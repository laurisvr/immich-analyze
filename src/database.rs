use crate::error::ImageAnalysisError;
use serde::Serialize;
use tokio_postgres::Client as PgClient;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ImageAnalysisResult {
    pub description: String,
    pub asset_id: Uuid,
}

/// Check if asset already has description in database
pub async fn asset_has_description(
    client: &PgClient,
    asset_id: Uuid,
) -> Result<bool, ImageAnalysisError> {
    let query = "
        SELECT EXISTS (
            SELECT 1 FROM asset_exif 
            WHERE \"assetId\"::text = $1 
            AND description IS NOT NULL 
            AND description != ''
        )
    ";
    let asset_id_str = asset_id.to_string();
    match client.query_one(query, &[&asset_id_str]).await {
        Ok(row) => Ok(row.get(0)),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!("database.error_checking_description", error = e.to_string())
            );
            Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })
        }
    }
}

/// Check if an asset still exists in the asset table.
///
/// A preview file in `thumbs/` is not proof that the asset is still there:
/// Immich does not always remove the thumbnail when an asset is deleted, so
/// the filesystem scan keeps rediscovering orphaned previews on every pass.
pub async fn check_asset_exists(
    client: &PgClient,
    asset_id: Uuid,
) -> Result<bool, ImageAnalysisError> {
    let query = "SELECT EXISTS (SELECT 1 FROM asset WHERE id = $1::uuid)";
    let asset_id_str = asset_id.to_string();
    match client.query_one(query, &[&asset_id_str]).await {
        Ok(row) => Ok(row.get(0)),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!(
                    "database.asset_existence_check_error",
                    error = e.to_string()
                )
            );
            Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })
        }
    }
}

/// Update or create asset description in database
pub async fn update_or_create_asset_description(
    client: &PgClient,
    asset_id: Uuid,
    description: &str,
) -> Result<(), ImageAnalysisError> {
    let safe_description = description.replace("'", "''");
    let safe_asset_id = asset_id.to_string();
    println!(
        "{}",
        rust_i18n::t!("database.updating_asset", asset_id = asset_id.to_string())
    );
    let preview: String = description.chars().take(100).collect();
    println!(
        "{}",
        rust_i18n::t!(
            "database.description_length",
            length = description.len().to_string(),
            preview = preview
        )
    );

    let update_query = format!(
        r#"
        UPDATE asset_exif 
        SET description = E'{}', 
            "updatedAt" = NOW(),
            "updateId" = immich_uuid_v7()
        WHERE "assetId" = '{}'
        "#,
        safe_description, safe_asset_id
    );
    match client.execute(&update_query, &[]).await {
        Ok(rows_affected) => {
            if rows_affected > 0 {
                println!(
                    "{}",
                    rust_i18n::t!("database.update_success", asset_id = asset_id.to_string())
                );
                return Ok(());
            }
        }
        Err(e) => {
            eprintln!(
                "{}\n{}",
                rust_i18n::t!(
                    "database.update_error",
                    asset_id = asset_id.to_string(),
                    error = e.to_string()
                ),
                rust_i18n::t!("database.sql_query_details", query = update_query)
            );
            return Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            });
        }
    }

    let asset_exists_query = format!(
        r#"
        SELECT EXISTS (
            SELECT 1 FROM asset 
            WHERE id = '{}'
        )
        "#,
        safe_asset_id
    );
    let asset_exists = match client.query_one(&asset_exists_query, &[]).await {
        Ok(row) => row.get::<_, bool>(0),
        Err(e) => {
            eprintln!(
                "{}",
                rust_i18n::t!(
                    "database.asset_existence_check_error",
                    error = e.to_string()
                )
            );
            return Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            });
        }
    };
    if !asset_exists {
        eprintln!(
            "{}",
            rust_i18n::t!(
                "database.asset_not_in_table",
                asset_id = asset_id.to_string()
            )
        );
        return Err(ImageAnalysisError::DatabaseError {
            error: format!(
                "{}",
                rust_i18n::t!(
                    "database.asset_not_found_error",
                    asset_id = asset_id.to_string()
                )
            ),
        });
    }

    let insert_query = format!(
        r#"
        INSERT INTO asset_exif (
            "assetId", description, "updatedAt", "updateId"
        ) VALUES (
            '{}', E'{}', NOW(), immich_uuid_v7()
        )
        ON CONFLICT ("assetId") DO UPDATE 
        SET description = EXCLUDED.description,
            "updatedAt" = NOW(),
            "updateId" = immich_uuid_v7()
        "#,
        safe_asset_id, safe_description
    );

    match client.execute(&insert_query, &[]).await {
        Ok(_) => {
            println!(
                "{}",
                rust_i18n::t!("database.insert_success", asset_id = asset_id.to_string())
            );
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "{}\n{}",
                rust_i18n::t!(
                    "database.insert_error",
                    asset_id = asset_id.to_string(),
                    error = e.to_string()
                ),
                rust_i18n::t!("database.sql_query_details", query = insert_query)
            );
            Err(ImageAnalysisError::DatabaseError {
                error: e.to_string(),
            })
        }
    }
}

/// Validate that a string is a valid pgvector literal.
fn is_valid_embedding(embedding: &str) -> bool {
    embedding.starts_with('[')
        && embedding.ends_with(']')
        && embedding[1..embedding.len() - 1]
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '-' || c == ',' || c == 'e' || c == 'E' || c == '+')
}

/// Upsert a single tag embedding into the tag_search table.
pub async fn upsert_tag_embedding(
    client: &PgClient,
    asset_id: Uuid,
    tag: &str,
    embedding: &str,
) -> Result<(), ImageAnalysisError> {
    if !is_valid_embedding(embedding) {
        return Err(ImageAnalysisError::ClipEncodingError {
            error: "Invalid embedding format".to_string(),
        });
    }

    let safe_tag = tag.replace('\'', "''");
    let query = format!(
        r#"INSERT INTO tag_search ("assetId", tag, embedding) VALUES ('{}', '{}', '{}')
           ON CONFLICT ("assetId", tag) DO UPDATE SET embedding = EXCLUDED.embedding"#,
        asset_id, safe_tag, embedding
    );

    match client.execute(&query, &[]).await {
        Ok(_) => {
            log::debug!("Upserted tag '{}' embedding for asset: {}", tag, asset_id);
            Ok(())
        }
        Err(e) => {
            log::warn!(
                "Failed to upsert tag '{}' for {}: {}",
                tag, asset_id, e
            );
            Ok(()) // Non-fatal
        }
    }
}

/// Delete tag rows for an asset that are not in the new tag set, so
/// re-analysis replaces the old tags instead of accumulating them.
/// Runs before the per-tag upserts: embeddings are deterministic per tag
/// text, so rows kept for still-present tags remain valid even if their
/// re-encode later fails.
pub async fn delete_stale_tags(
    client: &PgClient,
    asset_id: Uuid,
    keep_tags: &[&str],
) -> Result<(), ImageAnalysisError> {
    let keep_array = if keep_tags.is_empty() {
        "ARRAY[]::text[]".to_string()
    } else {
        format!(
            "ARRAY[{}]",
            keep_tags
                .iter()
                .map(|t| format!("'{}'", t.replace('\'', "''")))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let query = format!(
        r#"DELETE FROM tag_search WHERE "assetId" = '{}' AND tag <> ALL ({})"#,
        asset_id, keep_array
    );

    match client.execute(&query, &[]).await {
        Ok(n) => {
            if n > 0 {
                log::debug!("Deleted {} stale tag(s) for asset: {}", n, asset_id);
            }
            Ok(())
        }
        Err(e) => {
            log::warn!("Failed to delete stale tags for {}: {}", asset_id, e);
            Ok(()) // Non-fatal
        }
    }
}

pub async fn check_database_connection(client: &PgClient) -> Result<bool, ImageAnalysisError> {
    let timeout_duration = std::time::Duration::from_secs(5);
    match tokio::time::timeout(timeout_duration, client.query("SELECT 1", &[])).await {
        Ok(Ok(_)) => {
            println!("{}", rust_i18n::t!("database.connection_success"));
            Ok(true)
        }
        Ok(Err(e)) => {
            eprintln!(
                "{}",
                rust_i18n::t!("error.database_query_failed", error = e.to_string())
            );
            Err(ImageAnalysisError::DatabaseError {
                error: format!(
                    "{}",
                    rust_i18n::t!("database.query_failed_error", error = e.to_string())
                ),
            })
        }
        Err(_) => {
            eprintln!("{}", rust_i18n::t!("error.database_timeout"));
            Err(ImageAnalysisError::DatabaseError {
                error: format!("{}", rust_i18n::t!("database.timeout_error")),
            })
        }
    }
}
