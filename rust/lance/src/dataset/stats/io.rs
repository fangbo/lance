// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Storage helpers and IO for column statistics under `_stats`.
//!
//! File naming:
//! - Fragment-level: `colstats_frag_{frag_id}_{field_id}.lance`
//! - Dataset-level:  `colstats_ds_{version_id}_{field_id}.lance`
//!
//! This module provides:
//! - Path resolution helpers
//! - Arrow schema builders for stats files
//! - Read/write helpers using Lance file reader/writer

use std::sync::Arc;

use arrow_array::RecordBatch;

use lance_core::Result;
use lance_encoding::decoder::DecoderPlugins;
use lance_file::reader::FileReader;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_io::scheduler::{FileScheduler, ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use object_store::path::Path;

use crate::Dataset;

/// Returns the `_stats` directory path for the given dataset.
pub(crate) fn colstats_dir(dataset: &Dataset) -> Path {
    // Use base.child("_stats") similar to indices_dir/data_dir
    dataset.base.child("_stats")
}

/// Resolves fragment-level column statistics file path.
/// Format: `_stats/colstats_frag_{frag_id}_{field_id}.lance`
pub(crate) fn fragment_colstats_path(dataset: &Dataset, frag_id: u64, field_id: u32) -> Path {
    let filename = format!("colstats_frag_{}_{}.lance", frag_id, field_id);
    colstats_dir(dataset).child(filename.as_str())
}

/// Resolves dataset-level column statistics file path.
/// Format: `_stats/colstats_ds_{version_id}_{field_id}.lance`
pub(crate) fn dataset_colstats_path(dataset: &Dataset, version_id: u64, field_id: u32) -> Path {
    let filename = format!("colstats_ds_{}_{}.lance", version_id, field_id);
    colstats_dir(dataset).child(filename.as_str())
}

/// Write a fragment-level statistics record batch to `_stats`.
pub(crate) async fn write_fragment_colstats(
    dataset: &Dataset,
    frag_id: u64,
    _field_id: u32,
    rb: RecordBatch,
) -> Result<()> {
    let path = fragment_colstats_path(dataset, frag_id, _field_id);
    let schema = lance_core::datatypes::Schema::try_from(rb.schema().as_ref())?;
    let opts = FileWriterOptions {
        format_version: Some(dataset.manifest.data_storage_format.lance_file_version()?),
        ..Default::default()
    };
    // Single-batch write
    let batches = std::iter::once(rb);
    FileWriter::create_file_with_batches(&dataset.object_store, &path, schema, batches, opts)
        .await?;
    Ok(())
}

/// Read a fragment-level statistics record batch from `_stats`.
pub(crate) async fn read_fragment_colstats(
    dataset: &Dataset,
    frag_id: u64,
    field_id: u32,
) -> Result<Option<RecordBatch>> {
    let path = fragment_colstats_path(dataset, frag_id, field_id);
    if !dataset.object_store.exists(&path).await? {
        return Ok(None);
    }
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler: FileScheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await?;
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        dataset.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;

    // Read entire file using base projection
    let stream = reader.read_stream(
        lance_io::ReadBatchParams::RangeFull,
        1024,
        1,
        lance_encoding::decoder::FilterExpression::no_filter(),
    )?;

    // Collect all batches and concatenate (stats files are small)
    use futures::StreamExt;
    let mut batches = Vec::new();
    futures::pin_mut!(stream);
    while let Some(next) = stream.next().await {
        batches.push(next?);
    }
    if batches.is_empty() {
        return Ok(None);
    }
    let schema = batches[0].schema();
    let rb = arrow::compute::concat_batches(&schema, &batches)?;
    Ok(Some(rb))
}

/// Write a dataset-level statistics record batch to `_stats`.
pub(crate) async fn write_dataset_colstats(
    dataset: &Dataset,
    version_id: u64,
    field_id: u32,
    rb: RecordBatch,
) -> Result<()> {
    let path = dataset_colstats_path(dataset, version_id, field_id);
    let schema = lance_core::datatypes::Schema::try_from(rb.schema().as_ref())?;
    let opts = FileWriterOptions {
        format_version: Some(dataset.manifest.data_storage_format.lance_file_version()?),
        ..Default::default()
    };
    let batches = std::iter::once(rb);
    FileWriter::create_file_with_batches(&dataset.object_store, &path, schema, batches, opts)
        .await?;
    Ok(())
}

/// Read a dataset-level statistics record batch from `_stats`.
pub(crate) async fn read_dataset_colstats(
    dataset: &Dataset,
    version_id: u64,
    field_id: u32,
) -> Result<Option<RecordBatch>> {
    let path = dataset_colstats_path(dataset, version_id, field_id);
    if !dataset.object_store.exists(&path).await? {
        return Ok(None);
    }
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler: FileScheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await?;
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        dataset.file_reader_options.clone().unwrap_or_default(),
    )
    .await?;

    let stream = reader.read_stream(
        lance_io::ReadBatchParams::RangeFull,
        1024,
        1,
        lance_encoding::decoder::FilterExpression::no_filter(),
    )?;

    use futures::StreamExt;
    let mut batches = Vec::new();
    futures::pin_mut!(stream);
    while let Some(next) = stream.next().await {
        batches.push(next?);
    }
    if batches.is_empty() {
        return Ok(None);
    }
    let schema = batches[0].schema();
    let rb = arrow::compute::concat_batches(&schema, &batches)?;
    Ok(Some(rb))
}
