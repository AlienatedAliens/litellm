//! What the gateway knows of one call and how a callback reads and writes it by contract field,
//! the native counterpart of the proxy's `v2_proxy_fields.py`. A field the gateway cannot work
//! out yet is left out of what a callback is handed, and one it has no place for is not written.

use std::time::{SystemTime, UNIX_EPOCH};

use jiff::Timestamp;
use litellm_callbacks::{
    AttemptTiming, CallFacts, CallIds, CallKind, CallTarget, CallTiming, Deployment, Field,
    FieldSet, FieldValues, Hook, HookInput, KeyIdentity, LoggedKeyIdentity, LoggedModelGroup,
    MetadataKind, ModelInfo, Moment, Outcome, PatchOp, RateLimitState, Scalar, SpendLogId, Usage,
    UsageRecord,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, StoredKey};
use litellm_gateway_inference::{CallType, ServedCall};
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use serde::Serialize;
use serde_json::{Map, Value};

use super::parallel_request_limiter::{MASTER_KEY_ALIAS, PROXY_ADMIN_NAME};
use crate::host::NativeRows;

const MODEL: &str = "model";
const STREAM: &str = "stream";

/// A patch operation the gateway has no place to write.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum Unwritable {
    #[error("the gateway has no place to write {0:?} yet")]
    Field(Field),
    #[error("{0:?} is replaced whole, never merged into")]
    Merge(Field),
    #[error("{0:?} takes keys merged into it, never a whole value")]
    Set(Field),
    #[error("a model is a name, never {0}")]
    Model(Value),
    #[error("{0:?} is part of the request body, which the route already has")]
    Routed(Field),
}

/// The request body of one call. The route owns it, so it is only held while the pre-call hook
/// runs, and what it named of the call is all that stays once it is handed back. The model and
/// the stream flag are read from whichever is there, so a patch to a held body is what every
/// later reader sees.
enum Request {
    Held(Map<String, Value>),
    Named { model: Option<String>, stream: bool },
}

impl Request {
    fn body(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Held(body) => Some(body),
            Self::Named { .. } => None,
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            Self::Held(body) => body.get(MODEL).and_then(Value::as_str),
            Self::Named { model, .. } => model.as_deref(),
        }
    }

    fn stream(&self) -> bool {
        match self {
            Self::Held(body) => body.get(STREAM).and_then(Value::as_bool) == Some(true),
            Self::Named { stream, .. } => *stream,
        }
    }

    fn named(&self) -> Self {
        Self::Named {
            model: self.model().map(str::to_owned),
            stream: self.stream(),
        }
    }

    /// The held body for a write to `field`.
    fn held(&mut self, field: Field) -> Result<&mut Map<String, Value>, Unwritable> {
        match self {
            Self::Held(body) => Ok(body),
            Self::Named { .. } => Err(Unwritable::Routed(field)),
        }
    }
}

/// Who a call is charged to, as Python's `UserAPIKeyAuth` names its caller: a key by the hash
/// its row is stored under, its alias and its owner, and the master key by Python's alias for it
/// and the proxy admin. Native keys name no team, organization, project or end user.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Charged {
    pub(super) key_hash: Option<String>,
    pub(super) key_alias: Option<String>,
    pub(super) user_id: Option<String>,
}

impl Charged {
    pub(super) fn of(identity: &Authenticated<NativeRows>) -> Self {
        match &**identity {
            Caller::MasterKey => Self {
                key_hash: Some(MASTER_KEY_ALIAS.to_owned()),
                key_alias: None,
                user_id: Some(PROXY_ADMIN_NAME.to_owned()),
            },
            Caller::Key(key) => Self {
                key_hash: identity.hash().map(ToString::to_string),
                key_alias: key.key.alias().map(str::to_owned),
                user_id: key.key.record().user_id.clone(),
            },
        }
    }
}

/// The route a call came in on, as Python's `user_api_key_request_route` names it.
fn route(call_type: CallType) -> &'static str {
    match call_type {
        CallType::AnthropicMessages => "/v1/messages",
        CallType::Aocr => "/v1/ocr",
    }
}

/// One call as it stands.
pub(super) struct Known {
    call_type: CallType,
    call_id: Option<String>,
    charged: Charged,
    start: SystemTime,
    /// When the pre-call hook handed the body back for routing, Python's `api_call_start_time`.
    api_call_start: Option<SystemTime>,
    end: Option<SystemTime>,
    request: Request,
    rate_limit_statuses: Option<Value>,
    response_headers: Map<String, Value>,
    served: Option<ServedCall>,
    outcome: Option<Outcome>,
}

fn unix(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// `at` as the UTC `datetime` Python logs it as.
fn moment(at: SystemTime) -> Option<Moment> {
    let utc = Timestamp::try_from(at)
        .ok()?
        .to_zoned(jiff::tz::TimeZone::UTC);
    Some(Moment {
        year: i32::from(utc.year()),
        month: u8::try_from(utc.month()).ok()?,
        day: u8::try_from(utc.day()).ok()?,
        hour: u8::try_from(utc.hour()).ok()?,
        minute: u8::try_from(utc.minute()).ok()?,
        second: u8::try_from(utc.second()).ok()?,
        microsecond: u32::try_from(utc.subsec_nanosecond() / 1_000).ok()?,
        utc_offset_seconds: Some(0),
    })
}

fn text(value: Option<&str>) -> Option<Scalar> {
    value.map(|value| Scalar::Text(value.to_owned()))
}

fn record(value: impl Serialize) -> Option<Value> {
    serde_json::to_value(value).ok()
}

/// The headers the proxy's limiters add to an answer.
const LIMITER_HEADER_PREFIX: &str = "x-ratelimit-";

/// A header value as the hidden params hold it, which the limiter writes as a number or as text.
fn header_scalar(value: &Value) -> Option<Scalar> {
    match value {
        Value::Bool(flag) => Some(Scalar::Bool(*flag)),
        Value::Number(number) => number
            .as_i64()
            .map(|whole| Scalar::Int(i128::from(whole)))
            .or_else(|| number.as_f64().map(Scalar::Float)),
        Value::String(text) => Some(Scalar::Text(text.clone())),
        _ => None,
    }
}

fn count(tokens: u64) -> Option<Scalar> {
    Some(Scalar::Int(i128::from(tokens)))
}

/// A served call's usage as the success log's `response_obj` carries it, and as its logging
/// payload counts it, which holds the same counts.
fn usage(used: &ChatCompletionsUsage) -> Usage {
    let record = UsageRecord {
        prompt_tokens: count(used.prompt_tokens),
        completion_tokens: count(used.completion_tokens),
        total_tokens: count(used.total_tokens),
        cached_tokens: count(used.prompt_tokens_details.cached_tokens),
        ..UsageRecord::default()
    };
    Usage {
        response: Some(record.clone()),
        logged: Some(record),
        ..Usage::default()
    }
}

impl Known {
    pub(super) fn of(call_type: CallType, charged: Charged, call_id: Option<String>) -> Self {
        Self {
            call_type,
            call_id,
            charged,
            start: SystemTime::now(),
            api_call_start: None,
            end: None,
            request: Request::Named {
                model: None,
                stream: false,
            },
            rate_limit_statuses: None,
            response_headers: Map::new(),
            served: None,
            outcome: None,
        }
    }

    /// Takes the route's `body` for the pre-call hook.
    pub(super) fn hold(&mut self, body: Map<String, Value>) {
        self.request = Request::Held(body);
    }

    /// The body as the callbacks left it, back to the route, which calls the provider with it
    /// next.
    pub(super) fn release(&mut self) -> Map<String, Value> {
        self.api_call_start = Some(SystemTime::now());
        let named = self.request.named();
        match std::mem::replace(&mut self.request, named) {
            Request::Held(body) => body,
            Request::Named { .. } => Map::new(),
        }
    }

    pub(super) fn serve(&mut self, served: ServedCall) {
        self.end = Some(SystemTime::now());
        self.served = Some(served);
        self.outcome = Some(Outcome::Success);
    }

    pub(super) fn fail(&mut self) {
        self.end = Some(SystemTime::now());
        self.outcome = Some(Outcome::Failure);
    }

    fn deployment(&self) -> Option<&litellm_router_core::Deployment> {
        self.served.as_ref()?.deployment.as_deref()
    }

    /// The provider the served deployment was called through, as Python's
    /// `custom_llm_provider` names it.
    fn provider(&self) -> Option<String> {
        let deployment = self.deployment()?;
        deployment
            .resolved_provider
            .clone()
            .or_else(|| deployment.params.custom_llm_provider.clone())
            .or_else(|| {
                deployment
                    .params
                    .model
                    .split_once('/')
                    .map(|(provider, _)| provider.to_owned())
            })
    }

    fn key_identity(&self) -> KeyIdentity {
        let charged = &self.charged;
        KeyIdentity {
            key_hash: text(charged.key_hash.as_deref()),
            api_key: text(charged.key_hash.as_deref()),
            key_alias: text(charged.key_alias.as_deref()),
            user_id: text(charged.user_id.as_deref()),
            ..KeyIdentity::default()
        }
    }

    fn logged_key_identity(&self) -> LoggedKeyIdentity {
        let charged = &self.charged;
        LoggedKeyIdentity {
            metadata_kind: Some(MetadataKind::Mapping),
            key_hash: text(charged.key_hash.as_deref()),
            key_alias: text(charged.key_alias.as_deref()),
            user_id: text(charged.user_id.as_deref()),
            request_route: text(Some(route(self.call_type))),
            ..LoggedKeyIdentity::default()
        }
    }

    fn served_deployment(&self) -> Option<Deployment> {
        let deployment = self.deployment()?;
        let rate = |name: &str| match deployment.raw_params.get(name)? {
            Value::Number(number) => Some(number.as_i64().map_or_else(
                || Scalar::Float(number.as_f64().unwrap_or(0.0)),
                |whole| Scalar::Int(i128::from(whole)),
            )),
            Value::String(spelled) => Some(Scalar::Text(spelled.clone())),
            _ => None,
        };
        Some(Deployment {
            model_info: Some(ModelInfo {
                id: text(Some(&deployment.id)),
                tpm: rate("tpm"),
                rpm: rate("rpm"),
                ..ModelInfo::default()
            }),
            provider: text(self.provider().as_deref()),
            tpm: rate("tpm"),
            rpm: rate("rpm"),
            params_present: Some(true),
            ..Deployment::default()
        })
    }

    /// The id the call's spend log row is written under, as `get_spend_logs_id` picks it: the
    /// provider's id for the answer, else the call id.
    /// What the logging handlers read of the limits a served call is under: its model group, and
    /// the limiter's `x-ratelimit-*` headers among the answer's headers, which the native answer
    /// always carries. The provider's own remaining counts and a key's per-model remaining counts
    /// are not told by the native route yet.
    fn rate_limit_state(&self) -> Option<RateLimitState> {
        let served = self.served.as_ref()?;
        let limiter_headers = self
            .response_headers
            .iter()
            .filter(|(name, _)| name.starts_with(LIMITER_HEADER_PREFIX))
            .filter_map(|(name, value)| Some((name.clone(), header_scalar(value)?)))
            .collect();
        Some(RateLimitState {
            key_model_group: Some(Scalar::Text(served.model_group.clone())),
            limiter_headers: Some(limiter_headers),
            hidden_params_present: Some(true),
            additional_headers_present: Some(true),
            ..RateLimitState::default()
        })
    }

    fn spend_log_id(&self) -> SpendLogId {
        let answered = self
            .served
            .as_ref()
            .and_then(|served| served.response_id.clone());
        SpendLogId {
            id: answered.or_else(|| self.call_id.clone()),
            ..SpendLogId::default()
        }
    }

    /// The call's times. A complete answer started its completion when it ended, as Python
    /// defaults `completion_start_time` to the end.
    fn timing(&self) -> AttemptTiming {
        let first_event = self.served.as_ref().and_then(|served| served.first_event);
        let completion_start = first_event.or(self.end);
        AttemptTiming {
            start: moment(self.start),
            end: self.end.and_then(moment),
            api_call_start: self.api_call_start.and_then(moment),
            completion_start: completion_start.and_then(moment),
            completion_start_present: Some(completion_start.is_some()),
            logged_start: moment(self.start),
            logged_end: self.end.and_then(moment),
            ..AttemptTiming::default()
        }
    }

    pub(super) fn response_headers(&self) -> &Map<String, Value> {
        &self.response_headers
    }

    fn read(&self, field: Field) -> Option<Value> {
        match field {
            Field::Model => self.request.model().map(Value::from),
            Field::RequestBody => self.request.body().cloned().map(Value::Object),
            Field::RateLimitStatuses => self.rate_limit_statuses.clone(),
            Field::ResponseHeaders => Some(Value::Object(self.response_headers.clone())),
            Field::Outcome => record(self.outcome?),
            Field::Usage => record(usage(self.served.as_ref()?.usage.as_ref()?)),
            Field::KeyIdentity => record(self.key_identity()),
            Field::LoggedKeyIdentity => record(self.logged_key_identity()),
            Field::Deployment => record(self.served_deployment()?),
            Field::AttemptTiming => record(self.timing()),
            Field::SpendLogId => record(self.spend_log_id()),
            Field::RateLimitState => record(self.rate_limit_state()?),
            // `admit` leaves a body that names tags, and a key that holds any, to Python.
            Field::RequestTags => Some(Value::Array(Vec::new())),
            Field::LoggedModelGroup => record(LoggedModelGroup {
                value: Some(Scalar::Text(self.served.as_ref()?.model_group.clone())),
                ..LoggedModelGroup::default()
            }),
            _ => None,
        }
    }

    fn facts(&self) -> CallFacts {
        let deployment = self.deployment();
        let first_event = self.served.as_ref().and_then(|served| served.first_event);
        CallFacts {
            ids: CallIds {
                call_id: self.call_id.clone().unwrap_or_default(),
                ..CallIds::default()
            },
            kind: CallKind {
                call_type: <&str>::from(self.call_type).to_owned(),
                stream: self.request.stream(),
                ..CallKind::default()
            },
            target: CallTarget {
                model: self.request.model().unwrap_or_default().to_owned(),
                model_group: self
                    .served
                    .as_ref()
                    .map(|served| served.model_group.clone()),
                provider: self.provider(),
                api_base: deployment.and_then(|deployment| deployment.params.api_base.clone()),
                deployment_id: deployment.map(|deployment| deployment.id.clone()),
                ..CallTarget::default()
            },
            timing: CallTiming {
                start: unix(self.start),
                provider_start: self.api_call_start.map(unix),
                first_token: first_event.or(self.end).map(unix),
                end: self.end.map(unix),
            },
            cache_hit: None,
        }
    }

    /// What a callback that reads `reads` is handed on `hook`.
    pub(super) fn input(&self, hook: Hook, reads: FieldSet) -> HookInput {
        let fields = reads
            .iter()
            .filter(|field| !field.by_reference())
            .filter_map(|field| Some((field, self.read(field)?)))
            .fold(FieldValues::new(reads), |fields, (field, value)| {
                fields.with(field, value)
            });
        HookInput {
            hook,
            fields,
            call: Some(self.facts()),
        }
    }

    /// Writes one operation of a callback's patch: the request's own params into the body, as
    /// the proxy's hooks change `data`, and the rest where a later callback or the route reads it.
    pub(super) fn write(&mut self, op: PatchOp) -> Result<(), Unwritable> {
        match op {
            PatchOp::Merge {
                field: Field::OptionalParams,
                keys,
            } => self.request.held(Field::OptionalParams)?.extend(keys),
            PatchOp::Merge {
                field: Field::ResponseHeaders,
                keys,
            } => self.response_headers.extend(keys),
            PatchOp::Set {
                field: Field::Model,
                value: Value::String(model),
            } => {
                self.request
                    .held(Field::Model)?
                    .insert(MODEL.to_owned(), Value::from(model));
            }
            PatchOp::Set {
                field: Field::Model,
                value,
            } => return Err(Unwritable::Model(value)),
            PatchOp::Set {
                field: Field::RateLimitStatuses,
                value,
            } => self.rate_limit_statuses = Some(value),
            PatchOp::Set {
                field: field @ (Field::OptionalParams | Field::ResponseHeaders),
                ..
            } => return Err(Unwritable::Set(field)),
            PatchOp::Merge {
                field: field @ (Field::Model | Field::RateLimitStatuses),
                ..
            } => return Err(Unwritable::Merge(field)),
            PatchOp::Set { field, .. } | PatchOp::Merge { field, .. } => {
                return Err(Unwritable::Field(field));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn known() -> Known {
        let mut known = Known::of(CallType::AnthropicMessages, Charged::default(), None);
        known.hold(Map::from_iter([("model".to_owned(), json!("claude"))]));
        known
    }

    fn keys() -> Map<String, Value> {
        Map::from_iter([("max_tokens".to_owned(), json!(5))])
    }

    #[rstest]
    #[case::a_field_with_no_place(
        PatchOp::Set { field: Field::Metadata, value: json!({}) },
        Unwritable::Field(Field::Metadata)
    )]
    #[case::params_replaced_whole(
        PatchOp::Set { field: Field::OptionalParams, value: json!({}) },
        Unwritable::Set(Field::OptionalParams)
    )]
    #[case::a_model_that_is_no_name(
        PatchOp::Set { field: Field::Model, value: json!(7) },
        Unwritable::Model(json!(7))
    )]
    #[case::keys_merged_into_a_model(
        PatchOp::Merge { field: Field::Model, keys: keys() },
        Unwritable::Merge(Field::Model)
    )]
    fn a_write_the_gateway_has_no_place_for_leaves_the_call_as_it_was(
        #[case] op: PatchOp,
        #[case] refused: Unwritable,
    ) {
        let mut known = known();
        assert_eq!(
            (known.write(op), known.release()),
            (
                Err(refused),
                Map::from_iter([("model".to_owned(), json!("claude"))])
            )
        );
    }

    /// What a callback that reads the model is told of the call: the field, and the model and
    /// the stream flag among the call's facts.
    fn told(known: &Known, hook: Hook) -> (Option<Value>, String, bool) {
        let input = known.input(hook, FieldSet::of(&[Field::Model]));
        let call = input.call.unwrap();
        (
            input.fields.raw(Field::Model).unwrap().cloned(),
            call.target.model,
            call.kind.stream,
        )
    }

    #[rstest]
    #[case::the_model_set(
        PatchOp::Set { field: Field::Model, value: json!("claude-fallback") },
        (Some(json!("claude-fallback")), "claude-fallback".to_owned(), false)
    )]
    #[case::params_that_name_a_model_and_a_stream(
        PatchOp::Merge {
            field: Field::OptionalParams,
            keys: Map::from_iter([
                ("model".to_owned(), json!("claude-fallback")),
                ("stream".to_owned(), json!(true)),
            ]),
        },
        (Some(json!("claude-fallback")), "claude-fallback".to_owned(), true)
    )]
    fn what_a_callback_changes_of_the_body_is_what_the_next_one_and_the_route_see(
        #[case] op: PatchOp,
        #[case] seen: (Option<Value>, String, bool),
    ) {
        let mut known = known();
        known.write(op).unwrap();
        let on_the_same_hook = told(&known, Hook::PreCall);
        let body = known.release();
        assert_eq!(
            (
                on_the_same_hook,
                body.get(MODEL).cloned(),
                told(&known, Hook::AttemptEnded)
            ),
            (seen.clone(), seen.0.clone(), seen)
        );
    }

    fn served(response_id: Option<&str>) -> ServedCall {
        ServedCall {
            model_group: "claude".to_owned(),
            deployment: None,
            response_id: response_id.map(str::to_owned),
            usage: None,
            first_event: None,
        }
    }

    #[rstest]
    #[case::the_answers_own_id(Some(served(Some("msg_1"))), Some("msg_1"))]
    #[case::the_call_id_for_an_answer_that_names_none(Some(served(None)), Some("call-1"))]
    #[case::the_call_id_for_a_failure(None, Some("call-1"))]
    fn a_spend_log_row_is_written_under_the_answers_id_or_else_the_call_id(
        #[case] served: Option<ServedCall>,
        #[case] id: Option<&str>,
    ) {
        let mut known = Known::of(
            CallType::AnthropicMessages,
            Charged::default(),
            Some("call-1".to_owned()),
        );
        match served {
            Some(served) => known.serve(served),
            None => known.fail(),
        }
        assert_eq!(known.spend_log_id().id.as_deref(), id);
    }

    #[rstest]
    fn a_body_the_route_already_has_is_never_written() {
        let mut known = known();
        known.release();
        assert_eq!(
            known.write(PatchOp::Merge {
                field: Field::OptionalParams,
                keys: keys()
            }),
            Err(Unwritable::Routed(Field::OptionalParams))
        );
    }
}
