//! Which requests count toward the gateway request totals, and the middleware that records each
//! one with the status it answered.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::Method,
    middleware::Next,
    response::Response,
};

use crate::gateway_requests::{Category, GatewayRequests};

/// `_LLM_ROUTE_SUFFIXES`, a longer suffix before a shorter one it ends with.
const LLM_ROUTE_SUFFIXES: [&str; 20] = [
    "/chat/completions",
    "/completions",
    "/embeddings",
    "/responses",
    "/rerank",
    "/moderations",
    "/images/generations",
    "/images/edits",
    "/images/variations",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/videos",
    "/remix",
    "/ocr",
    "/search",
    "/rag/query",
    "/rag/ingest",
    ":generateContent",
    ":streamGenerateContent",
];

/// `_LLM_ROUTE_EXACT`.
const LLM_ROUTE_EXACT: [&str; 5] = [
    "/v1/messages",
    "/interactions",
    "/v1beta/interactions",
    "/comprehendmedical",
    "/transcribe",
];

const OPENSEARCH: &str = "/opensearch";

/// `LiteLLMRoutes.mapped_pass_through_routes` without `/langfuse` and `/opensearch`, which never
/// count as provider pass-through.
const PASSTHROUGH_PREFIXES: [&str; 31] = [
    "/backend-api/codex",
    "/bedrock",
    "/comprehendmedical",
    "/azure_speech",
    "/transcribe",
    "/vertex-ai",
    "/vertex_ai",
    "/cohere",
    "/cursor",
    "/cursor-cli",
    "/gemini",
    "/anthropic",
    "/azure",
    "/azure_ai",
    "/openai",
    "/openai_passthrough",
    "/assemblyai",
    "/eu.assemblyai",
    "/tinyfish",
    "/vllm",
    "/mistral",
    "/typesafe",
    "/laya",
    "/bespoke",
    "/openrouter",
    "/milvus",
    "/gigachat",
    "/watsonx",
    "/nvidia_nim",
    "/deepgram",
    "/fal_ai",
];

const MCP_MANAGEMENT_PREFIX: &str = "/v1/mcp";
const MCP_REST_TOOL_CALL: &str = "/mcp-rest/tools/call";
const A2A_INVOKE_SUFFIX: &str = "/message/send";
const A2A_TRANSPORT_PREFIXES: [&str; 2] = ["/v1/a2a/", "/a2a/"];

/// `path` past `prefix` and the slash after it.
fn below<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix)?.strip_prefix('/')
}

fn is_segment(text: &str) -> bool {
    !text.is_empty() && !text.contains('/')
}

/// `_MCP_DYNAMIC_TRANSPORT`, `/(?:toolset/)?[^/]+/mcp` matched whole.
fn is_dynamic_mcp_transport(path: &str) -> bool {
    path.strip_prefix('/')
        .and_then(|path| path.strip_suffix("/mcp"))
        .is_some_and(|server| {
            is_segment(server) || server.strip_prefix("toolset/").is_some_and(is_segment)
        })
}

/// `_classify_mcp_route`.
fn mcp_route(path: &str) -> Option<&'static str> {
    if path == MCP_MANAGEMENT_PREFIX || below(path, MCP_MANAGEMENT_PREFIX).is_some() {
        return None;
    }
    (path == "/mcp"
        || below(path, "/mcp").is_some()
        || path == MCP_REST_TOOL_CALL
        || is_dynamic_mcp_transport(path))
    .then_some("/mcp")
}

/// `_classify_a2a_route`.
fn a2a_route(path: &str) -> Option<&'static str> {
    (path.ends_with(A2A_INVOKE_SUFFIX)
        && A2A_TRANSPORT_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix)))
    .then_some("/a2a")
}

/// `_classify_llm_route`, where OpenSearch counts only an index search.
fn llm_route(path: &str) -> Option<&'static str> {
    if let Some(rest) = below(path, OPENSEARCH) {
        let index_search = rest.strip_suffix("/_search").is_some_and(is_segment);
        return index_search.then_some(OPENSEARCH);
    }
    LLM_ROUTE_EXACT
        .into_iter()
        .find(|route| path == *route)
        .or_else(|| {
            LLM_ROUTE_SUFFIXES
                .into_iter()
                .find(|suffix| path.ends_with(suffix))
        })
        .or_else(|| {
            PASSTHROUGH_PREFIXES
                .into_iter()
                .find(|prefix| below(path, prefix).is_some())
        })
}

/// `classify_billable_request`: the category and the route a request counts under, from a closed
/// list rather than the raw path, or `None` when it does not count. MCP and A2A count on any
/// method, an inference route only on `POST`.
fn classify(path: &str, method: &Method) -> Option<(Category, &'static str)> {
    let path = match path.trim_end_matches('/') {
        "" => "/",
        trimmed => trimmed,
    };
    if let Some(route) = mcp_route(path) {
        return Some((Category::Mcp, route));
    }
    if let Some(route) = a2a_route(path) {
        return Some((Category::A2a, route));
    }
    if *method != Method::POST {
        return None;
    }
    llm_route(path).map(|route| (Category::Llm, route))
}

/// `BillableRequestMetricsMiddleware` with only its gateway request sink: a request that counts is
/// recorded with the status its response answered, a 2xx as successful and anything else as
/// failed. A handler that panics records nothing, as an exception that escapes the app records
/// nothing in Python. The gateway installs it only with a database, as Python builds the sink only
/// with one.
///
/// Accepted differences: Python records once the body is sent and this records when the response
/// starts, which moves only the flush a long stream lands in. The built-in EU JEV decision route
/// is not classified, since it counts in Python only while no pass-through endpoint claims it and
/// the gateway never serves it. The enterprise billing recorder is not ported.
pub(super) async fn count(
    State(requests): State<Arc<GatewayRequests>>,
    method: Method,
    request: Request,
    next: Next,
) -> Response {
    let Some((category, route)) = classify(request.uri().path(), &method) else {
        return next.run(request).await;
    };
    let response = next.run(request).await;
    requests.record(category, route, response.status());
    response
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, http::StatusCode, routing::any};
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    #[rstest]
    #[case::messages("/v1/messages", Some((Category::Llm, "/v1/messages")))]
    #[case::messages_trailing_slash("/v1/messages/", Some((Category::Llm, "/v1/messages")))]
    #[case::ocr("/v1/ocr", Some((Category::Llm, "/ocr")))]
    #[case::bare_ocr("/ocr", Some((Category::Llm, "/ocr")))]
    #[case::chat_before_completions("/v1/chat/completions", Some((Category::Llm, "/chat/completions")))]
    #[case::completions("/v1/completions", Some((Category::Llm, "/completions")))]
    #[case::gemini_native(
        "/v1beta/models/gemini-pro:generateContent",
        Some((Category::Llm, ":generateContent"))
    )]
    #[case::vector_store_search("/v1/vector_stores/vs_1/search", Some((Category::Llm, "/search")))]
    #[case::assistants_messages_do_not_count("/v1/threads/thread_1/messages", None)]
    #[case::interactions("/v1beta/interactions", Some((Category::Llm, "/v1beta/interactions")))]
    #[case::interaction_read_does_not_count("/v1beta/interactions/abc", None)]
    #[case::comprehend_medical("/comprehendmedical", Some((Category::Llm, "/comprehendmedical")))]
    #[case::bedrock_passthrough(
        "/bedrock/model/claude/invoke",
        Some((Category::Llm, "/bedrock"))
    )]
    #[case::bare_passthrough_prefix("/bedrock", None)]
    #[case::langfuse_does_not_count("/langfuse/api/public/ingestion", None)]
    #[case::opensearch_index_search("/opensearch/logs/_search", Some((Category::Llm, "/opensearch")))]
    #[case::opensearch_document_write("/opensearch/logs/_doc", None)]
    #[case::opensearch_deep_search("/opensearch/logs/extra/_search", None)]
    #[case::management("/key/generate", None)]
    #[case::root("/", None)]
    fn a_post_counts_only_on_an_inference_route(
        #[case] path: &str,
        #[case] expected: Option<(Category, &'static str)>,
    ) {
        assert_eq!(classify(path, &Method::POST), expected);
    }

    #[rstest]
    #[case::mcp("/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_below("/mcp/tools/list", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_rest_tool_call("/mcp-rest/tools/call", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_rest_tool_list("/mcp-rest/tools/list", None)]
    #[case::dynamic_server("/github/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::toolset("/toolset/dev/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::server_named_toolset("/toolset/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::too_deep("/a/b/c/mcp", None)]
    #[case::management("/v1/mcp/server", None)]
    #[case::a2a("/a2a/agent-1/message/send", Some((Category::A2a, "/a2a")))]
    #[case::a2a_v1("/v1/a2a/agent-1/message/send", Some((Category::A2a, "/a2a")))]
    #[case::bare_a2a("/a2a/agent-1", None)]
    #[case::inference_on_get("/v1/messages", None)]
    fn mcp_and_a2a_count_on_any_method(
        #[case] path: &str,
        #[case] expected: Option<(Category, &'static str)>,
    ) {
        assert_eq!(classify(path, &Method::GET), expected);
    }

    async fn counted(path: &str, status: StatusCode) -> String {
        let requests = Arc::new(GatewayRequests::new());
        let app = Router::new()
            .route("/{*path}", any(move || async move { status }))
            .layer(axum::middleware::from_fn_with_state(
                requests.clone(),
                count,
            ));
        let response = app
            .oneshot(Request::post(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        requests.summary()
    }

    #[rstest]
    #[case::success("/v1/messages", StatusCode::OK, "llm /v1/messages 200: 1 ok 0 failed")]
    #[case::failure(
        "/v1/messages",
        StatusCode::BAD_GATEWAY,
        "llm /v1/messages 502: 0 ok 1 failed"
    )]
    #[case::not_counted("/key/generate", StatusCode::OK, "")]
    #[tokio::test]
    async fn a_counted_request_is_recorded_with_its_status(
        #[case] path: &str,
        #[case] status: StatusCode,
        #[case] expected: &str,
    ) {
        assert_eq!(counted(path, status).await, expected);
    }
}
