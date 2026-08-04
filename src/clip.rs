//! Per-tag CLIP embeddings for hybrid semantic search.
//!
//! After the vision model produces a description, the trailing `Tags:` line is split
//! into individual tags and each one is embedded separately via the Immich ML
//! container's CLIP text encoder, then stored one row per tag in `tag_search`.
//!
//! Per-tag embeddings give strong, narrow matches: a query like "toad" scores 1.0
//! against its exact tag versus 0.06 when every tag is blended into one embedding.
//! `LABSE-Vit-L-14` is the recommended encoder — its cross-lingual text semantics are
//! far better (kikker/toad normalized: 0.87 vs 0.25 for MCLIP) while keeping garbage
//! matches near zero.

use crate::{
    config::ClipConfig, data_access::DataAccess, database::ImageAnalysisResult,
    error::ImageAnalysisError,
};
use log::warn;
use reqwest::Client;
use serde_json::Value;
use std::{sync::OnceLock, time::Duration};

static CLIP_CLIENT: OnceLock<Client> = OnceLock::new();

/// Dedicated HTTP client for the ML container.
///
/// Kept module-local instead of threaded through `ProcessingContext` so this fork adds
/// no parameters to upstream function signatures, which keeps the rebase surface small.
fn clip_client() -> &'static Client {
    CLIP_CLIENT.get_or_init(Client::new)
}

/// Extract the `Tags: ...` line from a description, if the model produced one.
///
/// This reads the raw model output, never the stored description, so the
/// `[AI]...[/AI]` wrapper applied at storage time is irrelevant here.
#[must_use]
pub fn extract_tags(description: &str) -> Option<String> {
    description.lines().find_map(|line| {
        let trimmed = line.trim();
        let tags = trimmed
            .strip_prefix("Tags:")
            .or_else(|| trimmed.strip_prefix("tags:"))?
            .trim();
        (!tags.is_empty()).then(|| tags.to_owned())
    })
}

/// Encode a single text string into a CLIP embedding vector.
async fn encode_single(
    config: &ClipConfig,
    url: &str,
    text: &str,
) -> Result<Vec<f64>, ImageAnalysisError> {
    let entries = serde_json::json!({
        "clip": {
            "textual": {
                "modelName": config.model_name
            }
        }
    });

    let form = reqwest::multipart::Form::new()
        .text("entries", entries.to_string())
        .text("text", text.to_owned());

    let response = tokio::time::timeout(
        Duration::from_secs(config.timeout),
        clip_client().post(url).multipart(form).send(),
    )
    .await
    .map_err(|_| ImageAnalysisError::ClipEncodingError {
        error: "CLIP encoding request timed out".to_owned(),
    })?
    .map_err(|err| ImageAnalysisError::ClipEncodingError {
        error: format!("HTTP error: {err}"),
    })?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(ImageAnalysisError::ClipEncodingError {
            error: format!("HTTP {status} from ML container: {body}"),
        });
    }

    let body: Value =
        response
            .json()
            .await
            .map_err(|err| ImageAnalysisError::ClipEncodingError {
                error: format!("Failed to parse ML response: {err}"),
            })?;

    let clip_val = body
        .get("clip")
        .ok_or_else(|| ImageAnalysisError::ClipEncodingError {
            error: "Missing 'clip' field in ML response".to_owned(),
        })?;

    // The ML container returns the embedding as a JSON string containing an array;
    // array-typed responses are handled as a fallback across container versions.
    clip_val.as_str().map_or_else(
        || {
            clip_val.as_array().map_or_else(
                || {
                    Err(ImageAnalysisError::ClipEncodingError {
                        error: "Unexpected type for 'clip' field in ML response".to_owned(),
                    })
                },
                |values| {
                    values
                        .iter()
                        .map(|val| {
                            val.as_f64()
                                .ok_or_else(|| ImageAnalysisError::ClipEncodingError {
                                    error: "Non-numeric value in embedding array".to_owned(),
                                })
                        })
                        .collect()
                },
            )
        },
        |encoded| {
            serde_json::from_str(encoded).map_err(|err| ImageAnalysisError::ClipEncodingError {
                error: format!("Failed to parse embedding string: {err}"),
            })
        },
    )
}

/// Encode text into a pgvector literal via the Immich ML container's `/predict` endpoint.
async fn encode_text(config: &ClipConfig, text: &str) -> Result<String, ImageAnalysisError> {
    let url = format!("{}/predict", config.url.trim_end_matches('/'));
    let embedding = encode_single(config, &url, text).await?;
    let parts: Vec<String> = embedding.iter().map(ToString::to_string).collect();
    Ok(format!("[{}]", parts.join(",")))
}

/// Embed each tag from an analysis result and store it in `tag_search`.
///
/// Entirely non-fatal: the description has already been written by the time this runs,
/// so a CLIP outage degrades tag search rather than failing the analysis.
pub async fn store_description_tags(
    data_access: &DataAccess,
    clip: Option<&ClipConfig>,
    analysis: &ImageAnalysisResult,
) {
    let Some(config) = clip else { return };
    let Some(tags_line) = extract_tags(&analysis.description) else {
        return;
    };
    let tags: Vec<&str> = tags_line
        .split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .collect();
    if tags.is_empty() {
        return;
    }

    // Drop tags this asset no longer has before re-encoding, so re-analysis replaces
    // the old tag set instead of accumulating it. Embeddings are deterministic per tag
    // text, so rows kept for still-present tags stay valid even if a re-encode fails.
    if let Err(err) = data_access
        .delete_stale_tags(&analysis.asset_id, &tags)
        .await
    {
        warn!(
            "Failed to delete stale tags for {}: {err} (non-fatal)",
            analysis.asset_id
        );
    }

    for tag in tags {
        match encode_text(config, tag).await {
            Ok(embedding) => {
                if let Err(err) = data_access
                    .upsert_tag_embedding(&analysis.asset_id, tag, &embedding)
                    .await
                {
                    warn!(
                        "Failed to store tag '{tag}' for {}: {err} (non-fatal)",
                        analysis.asset_id
                    );
                }
            }
            Err(err) => {
                warn!(
                    "Failed to encode tag '{tag}' for {}: {err} (non-fatal)",
                    analysis.asset_id
                );
            }
        }
    }
}
