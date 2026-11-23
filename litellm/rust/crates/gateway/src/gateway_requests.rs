//! The gateway request counts (successful gateway requests, SGR) the request volume middleware
//! records for every request the gateway serves itself, folded in memory and added onto
//! `LiteLLM_DailyGatewayRequests` and `LiteLLM_DailyGatewayFailedRequests` every flush interval,
//! as `litellm/proxy/db/gateway_request_tracking.py` does for the requests Python serves. In front
//! of Python each side counts only the requests it answers, so the tables add up both.
//!
//! The fold keeps no per-request item. Every dimension of its key is bounded: the UTC date, the
//! category, a route from the classifier's closed list and the status code. A flush writes its
//! snapshot in statements of at most [`MAX_ROWS_PER_STATEMENT`] rows and puts back what it could
//! not write, so the next flush retries it: at least once, never lost to a database blip.
//!
//! Accepted differences: the proxy's `use_redis_transaction_buffer` path, which pushes snapshots
//! to Redis for one lock-holding pod to write, is not ported, because a config that sets it is
//! answered by Python. `general_settings.proxy_batch_write_at` is not read for the same reason,
//! only `PROXY_BATCH_WRITE_AT`.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
    time::Duration,
};

use axum::http::StatusCode;
use jiff::Timestamp;
use litellm_config::Environment;
use litellm_repositories::{GatewayRequestCount, GatewayRequestRepository};
use parking_lot::Mutex;
use rand::Rng;
use tokio::{sync::watch, task::JoinHandle};

/// `GATEWAY_REQUESTS_MAX_ROWS_PER_UPSERT`.
const MAX_ROWS_PER_STATEMENT: usize = 1000;
/// `PROXY_BATCH_WRITE_AT`'s default.
const DEFAULT_BATCH_WRITE_SECONDS: u64 = 10;
/// `_BATCH_WRITE_JITTER_MAX_SECONDS`.
const MAX_JITTER_SECONDS: u64 = 5;

/// `BillableCategory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, strum::IntoStaticStr)]
pub enum Category {
    #[strum(serialize = "llm")]
    Llm,
    #[strum(serialize = "mcp")]
    Mcp,
    #[strum(serialize = "a2a")]
    A2a,
}

/// `GatewayRequestKey`, with the status code always known.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Key {
    date: String,
    category: Category,
    route: &'static str,
    status_code: u16,
}

/// `GatewayRequestCounts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    successful: i64,
    failed: i64,
}

type Snapshot = HashMap<Key, Counts>;
/// A route's rows on one date, which always share a statement.
type RouteRows = BTreeMap<(String, Category, &'static str), Vec<(Key, Counts)>>;

/// `GatewayRequestAccumulator`: the fold the middleware records into and each flush drains.
pub struct GatewayRequests {
    clock: fn() -> Timestamp,
    counts: Mutex<Snapshot>,
}

impl GatewayRequests {
    pub fn new() -> Self {
        Self::with_clock(Timestamp::now)
    }

    fn with_clock(clock: fn() -> Timestamp) -> Self {
        Self {
            clock,
            counts: Mutex::new(Snapshot::new()),
        }
    }

    /// One request to `route` that answered `status`, under today's UTC date. Only a 2xx succeeded.
    pub fn record(&self, category: Category, route: &'static str, status: StatusCode) {
        let key = Key {
            date: (self.clock)().strftime("%Y-%m-%d").to_string(),
            category,
            route,
            status_code: status.as_u16(),
        };
        let mut counts = self.counts.lock();
        let entry = counts.entry(key).or_default();
        match status.is_success() {
            true => entry.successful += 1,
            false => entry.failed += 1,
        }
    }

    /// What the fold holds, one `category route status: n ok m failed` line per key in key order.
    #[cfg(test)]
    pub(crate) fn summary(&self) -> String {
        let counts = self.counts.lock();
        let mut lines: Vec<String> = counts
            .iter()
            .map(|(key, counts)| {
                format!(
                    "{} {} {}: {} ok {} failed",
                    <&str>::from(key.category),
                    key.route,
                    key.status_code,
                    counts.successful,
                    counts.failed
                )
            })
            .collect();
        lines.sort();
        lines.join("\n")
    }

    fn drain(&self) -> Snapshot {
        std::mem::take(&mut *self.counts.lock())
    }

    /// Merges counts a flush could not write back into the fold, summed key by key, so the fold
    /// stays bounded however long the database is unreachable.
    fn restore(&self, snapshot: Snapshot) {
        let mut counts = self.counts.lock();
        for (key, restored) in snapshot {
            let entry = counts.entry(key).or_default();
            entry.successful += restored.successful;
            entry.failed += restored.failed;
        }
    }

    /// `flush_gateway_requests` without a Redis buffer: writes the drained snapshot and keeps
    /// what did not land for the next flush.
    pub async fn flush(&self, repository: &GatewayRequestRepository) {
        let unwritten = commit(repository, self.drain()).await;
        self.restore(unwritten);
    }
}

impl Default for GatewayRequests {
    fn default() -> Self {
        Self::new()
    }
}

/// `commit_gateway_requests_to_db`: the snapshot in bounded statements, returning the rows of the
/// statement that failed and of every one after it, since replaying the whole snapshot would count
/// the statements that landed twice.
async fn commit(repository: &GatewayRequestRepository, snapshot: Snapshot) -> Snapshot {
    let chunks = statement_chunks(snapshot);
    let total = chunks.len();
    for (index, chunk) in chunks.iter().enumerate() {
        let counts = chunk.iter().map(stored).collect::<Vec<_>>();
        if let Err(error) = repository.increment(&counts).await {
            let unwritten: Snapshot = chunks[index..].iter().flatten().cloned().collect();
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                statement = index + 1,
                statements = total,
                rows = unwritten.len(),
                "gateway request tracking statement failed, its rows and every later one go back for retry"
            );
            return unwritten;
        }
    }
    Snapshot::new()
}

/// `_statement_chunks`: rows in key order, cut into statements of at most
/// [`MAX_ROWS_PER_STATEMENT`] rows, where a route's rows on one date always share a statement, so
/// its total and its status code rows land together.
fn statement_chunks(snapshot: Snapshot) -> Vec<Vec<(Key, Counts)>> {
    let mut routes = RouteRows::new();
    for (key, counts) in snapshot {
        routes
            .entry((key.date.clone(), key.category, key.route))
            .or_default()
            .push((key, counts));
    }
    routes.into_values().fold(
        Vec::new(),
        |mut chunks: Vec<Vec<(Key, Counts)>>, mut rows| {
            rows.sort_by_key(|(key, _)| key.status_code);
            match chunks.last_mut() {
                Some(pending) if pending.len() + rows.len() <= MAX_ROWS_PER_STATEMENT => {
                    pending.extend(rows);
                }
                _ => chunks.push(rows),
            }
            chunks
        },
    )
}

fn stored((key, counts): &(Key, Counts)) -> GatewayRequestCount {
    GatewayRequestCount {
        date: key.date.clone(),
        category: <&str>::from(key.category).to_owned(),
        route: key.route.to_owned(),
        status_code: Some(i32::from(key.status_code)),
        successful_requests: counts.successful,
        failed_requests: counts.failed,
    }
}

/// The flush job `proxy_startup_event` schedules as `update_gateway_requests_job`: every
/// `PROXY_BATCH_WRITE_AT` seconds plus up to five seconds of jitter, picked once, and one last
/// flush on shutdown.
pub struct Flusher {
    requests: Arc<GatewayRequests>,
    repository: Arc<GatewayRequestRepository>,
    stop: watch::Sender<bool>,
    job: Mutex<Option<JoinHandle<()>>>,
}

impl Flusher {
    /// Starts flushing `requests` through `repository` on the current runtime.
    pub fn start(
        requests: Arc<GatewayRequests>,
        repository: GatewayRequestRepository,
        environment: &Environment,
    ) -> Self {
        let repository = Arc::new(repository);
        let interval = interval(
            environment,
            rand::thread_rng().gen_range(0..=MAX_JITTER_SECONDS),
        );
        let (stop, mut stopped) = watch::channel(false);
        let job = tokio::spawn({
            let requests = requests.clone();
            let repository = repository.clone();
            async move {
                loop {
                    tokio::select! {
                        () = tokio::time::sleep(interval) => requests.flush(&repository).await,
                        _ = stopped.changed() => return,
                    }
                }
            }
        });
        Self {
            requests,
            repository,
            stop,
            job: Mutex::new(Some(job)),
        }
    }

    /// The fold the request volume middleware records into.
    pub fn requests(&self) -> Arc<GatewayRequests> {
        self.requests.clone()
    }

    /// `_flush_request_accumulators` on shutdown: waits out a flush already running, then writes
    /// what is left.
    pub async fn close(&self) {
        let _ = self.stop.send(true);
        let job = self.job.lock().take();
        if let Some(job) = job {
            let _ = job.await;
        }
        self.requests.flush(&self.repository).await;
    }
}

/// `proxy_batch_write_at + random.randint(0, 5)`, at least a second so a zero cannot spin.
fn interval(environment: &Environment, jitter_seconds: u64) -> Duration {
    let seconds = environment
        .proxy_batch_write_at
        .unwrap_or(DEFAULT_BATCH_WRITE_SECONDS)
        .saturating_add(jitter_seconds);
    Duration::from_secs(seconds.max(1))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn noon() -> Timestamp {
        "2026-11-23T12:00:00Z".parse().unwrap()
    }

    fn key(route: &'static str, status_code: u16) -> Key {
        Key {
            date: "2026-11-23".to_owned(),
            category: Category::Llm,
            route,
            status_code,
        }
    }

    #[rstest]
    fn requests_fold_by_route_and_status_with_only_a_2xx_succeeding() {
        let requests = GatewayRequests::with_clock(noon);

        requests.record(Category::Llm, "/v1/messages", StatusCode::OK);
        requests.record(Category::Llm, "/v1/messages", StatusCode::OK);
        requests.record(Category::Llm, "/v1/messages", StatusCode::CREATED);
        requests.record(Category::Llm, "/v1/messages", StatusCode::TOO_MANY_REQUESTS);
        requests.record(Category::Llm, "/v1/messages", StatusCode::MULTIPLE_CHOICES);

        assert_eq!(
            requests.drain(),
            Snapshot::from([
                (
                    key("/v1/messages", 200),
                    Counts {
                        successful: 2,
                        failed: 0
                    }
                ),
                (
                    key("/v1/messages", 201),
                    Counts {
                        successful: 1,
                        failed: 0
                    }
                ),
                (
                    key("/v1/messages", 429),
                    Counts {
                        successful: 0,
                        failed: 1
                    }
                ),
                (
                    key("/v1/messages", 300),
                    Counts {
                        successful: 0,
                        failed: 1
                    }
                ),
            ])
        );
        assert_eq!(requests.drain(), Snapshot::new());
    }

    #[rstest]
    fn restored_counts_add_onto_what_arrived_since() {
        let requests = GatewayRequests::with_clock(noon);
        requests.record(Category::Llm, "/v1/messages", StatusCode::OK);

        requests.restore(Snapshot::from([
            (
                key("/v1/messages", 200),
                Counts {
                    successful: 3,
                    failed: 0,
                },
            ),
            (
                key("/ocr", 500),
                Counts {
                    successful: 0,
                    failed: 2,
                },
            ),
        ]));

        assert_eq!(
            requests.drain(),
            Snapshot::from([
                (
                    key("/v1/messages", 200),
                    Counts {
                        successful: 4,
                        failed: 0
                    }
                ),
                (
                    key("/ocr", 500),
                    Counts {
                        successful: 0,
                        failed: 2
                    }
                ),
            ])
        );
    }

    fn routes(count: usize, statuses: &[u16]) -> Snapshot {
        const ROUTES: [&str; 3] = ["/a", "/b", "/c"];
        ROUTES[..count]
            .iter()
            .flat_map(|route| {
                statuses.iter().map(|status| {
                    (
                        key(route, *status),
                        Counts {
                            successful: 1,
                            failed: 0,
                        },
                    )
                })
            })
            .collect()
    }

    fn shape(chunks: &[Vec<(Key, Counts)>]) -> Vec<Vec<(&'static str, u16)>> {
        chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|(key, _)| (key.route, key.status_code))
                    .collect()
            })
            .collect()
    }

    #[rstest]
    fn a_small_snapshot_is_one_statement_in_key_order() {
        assert_eq!(
            shape(&statement_chunks(routes(2, &[500, 200]))),
            vec![vec![("/a", 200), ("/a", 500), ("/b", 200), ("/b", 500)]]
        );
    }

    #[rstest]
    fn a_route_that_would_overflow_a_statement_starts_the_next_one() {
        let statuses: Vec<u16> = (100..100 + 400).collect();

        let chunks = statement_chunks(routes(3, &statuses));

        assert_eq!(
            chunks.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![800, 400]
        );
        assert_eq!(chunks[1][0].0.route, "/c");
    }

    #[rstest]
    #[case::default(None, 0, 10)]
    #[case::configured(Some(30), 0, 30)]
    #[case::jittered(None, 5, 15)]
    #[case::zero(Some(0), 0, 1)]
    fn the_flush_interval_is_the_batch_write_interval_plus_jitter(
        #[case] proxy_batch_write_at: Option<u64>,
        #[case] jitter_seconds: u64,
        #[case] expected_seconds: u64,
    ) {
        let environment = Environment {
            proxy_batch_write_at,
            ..Environment::default()
        };

        assert_eq!(
            interval(&environment, jitter_seconds),
            Duration::from_secs(expected_seconds)
        );
    }
}
