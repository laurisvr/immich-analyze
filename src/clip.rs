use crate::error::ImageAnalysisError;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

/// Encode a single text string into a CLIP embedding vector.
async fn encode_single(
    client: &Client,
    url: &str,
    clip_model: &str,
    text: &str,
    timeout: u64,
) -> Result<Vec<f64>, ImageAnalysisError> {
    let entries = serde_json::json!({
        "clip": {
            "textual": {
                "modelName": clip_model
            }
        }
    });

    let form = reqwest::multipart::Form::new()
        .text("entries", entries.to_string())
        .text("text", text.to_string());

    let response = tokio::time::timeout(
        Duration::from_secs(timeout),
        client.post(url).multipart(form).send(),
    )
    .await
    .map_err(|_| ImageAnalysisError::ClipEncodingError {
        error: "CLIP encoding request timed out".to_string(),
    })?
    .map_err(|e| ImageAnalysisError::ClipEncodingError {
        error: format!("HTTP error: {}", e),
    })?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(ImageAnalysisError::ClipEncodingError {
            error: format!("HTTP {} from ML container: {}", status, body),
        });
    }

    let body: Value = response.json().await.map_err(|e| {
        ImageAnalysisError::ClipEncodingError {
            error: format!("Failed to parse ML response: {}", e),
        }
    })?;

    let clip_val = body
        .get("clip")
        .ok_or_else(|| ImageAnalysisError::ClipEncodingError {
            error: "Missing 'clip' field in ML response".to_string(),
        })?;

    // ML container returns embedding as a JSON string containing an array,
    // but handle array-typed responses as a fallback.
    if let Some(s) = clip_val.as_str() {
        serde_json::from_str(s).map_err(|e| ImageAnalysisError::ClipEncodingError {
            error: format!("Failed to parse embedding string: {}", e),
        })
    } else if let Some(arr) = clip_val.as_array() {
        arr.iter()
            .map(|v| v.as_f64().ok_or_else(|| ImageAnalysisError::ClipEncodingError {
                error: "Non-numeric value in embedding array".to_string(),
            }))
            .collect()
    } else {
        Err(ImageAnalysisError::ClipEncodingError {
            error: format!("Unexpected clip field type: {}", clip_val),
        })
    }
}

/// Encode text into a CLIP embedding via the Immich ML container's /predict endpoint.
pub async fn encode_text(
    client: &Client,
    clip_url: &str,
    clip_model: &str,
    text: &str,
    timeout: u64,
) -> Result<String, ImageAnalysisError> {
    let url = format!("{}/predict", clip_url.trim_end_matches('/'));
    let emb = encode_single(client, &url, clip_model, text, timeout).await?;
    let parts: Vec<String> = emb.iter().map(|v| format!("{}", v)).collect();
    Ok(format!("[{}]", parts.join(",")))
}
