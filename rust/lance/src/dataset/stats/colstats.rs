// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::datatypes::{
    ArrowPrimitiveType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use arrow_array::{
    ArrayRef, BinaryArray, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, RecordBatch, StringArray,
    StringViewArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion::scalar::ScalarValue;
use datafusion_expr::Accumulator;
use lance_core::datatypes::{Field, Schema};
use lance_core::Result;

use super::io::{
    fragment_colstats_path, read_dataset_colstats, read_fragment_colstats, write_dataset_colstats,
    write_fragment_colstats,
};
use super::{ColumnStatistics, DatasetStatsExt, FragmentStatsExt};
use crate::dataset::fragment::{FileFragment, FragReadConfig};
use crate::dataset::stats::hyperloglog::HyperLogLog;
use crate::dataset::Dataset;

/// Count NaNs for known floating-point arrays
fn count_nans(array: &ArrayRef) -> u64 {
    match array.data_type() {
        DataType::Float16 => {
            let arr = array.as_any().downcast_ref::<Float16Array>().unwrap();
            arr.values().iter().filter(|&&x| x.is_nan()).count() as u64
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            arr.values().iter().filter(|&&x| x.is_nan()).count() as u64
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            arr.values().iter().filter(|&&x| x.is_nan()).count() as u64
        }
        _ => 0,
    }
}

// HyperLogLog single-accumulator state per field
#[derive(Debug)]
enum HLLState {
    U8(HyperLogLog<<UInt8Type as ArrowPrimitiveType>::Native>),
    U16(HyperLogLog<<UInt16Type as ArrowPrimitiveType>::Native>),
    U32(HyperLogLog<<UInt32Type as ArrowPrimitiveType>::Native>),
    U64(HyperLogLog<<UInt64Type as ArrowPrimitiveType>::Native>),
    I8(HyperLogLog<<Int8Type as ArrowPrimitiveType>::Native>),
    I16(HyperLogLog<<Int16Type as ArrowPrimitiveType>::Native>),
    I32(HyperLogLog<<Int32Type as ArrowPrimitiveType>::Native>),
    I64(HyperLogLog<<Int64Type as ArrowPrimitiveType>::Native>),
    Str(HyperLogLog<String>),
    LStr(HyperLogLog<String>),
    Bin(HyperLogLog<Vec<u8>>),
    LBin(HyperLogLog<Vec<u8>>),
}

macro_rules! hll_extend_numeric {
    ($state:expr, $array:expr, $variant:ident, $arrty:ty) => {{
        if let HLLState::$variant(ref mut hll) = $state {
            let arr = $array.as_any().downcast_ref::<$arrty>().unwrap();
            hll.extend(arr.into_iter().flatten());
        }
    }};
}

macro_rules! hll_extend_string {
    ($state:expr, $array:expr, $variant:ident, $arrty:ty) => {{
        if let HLLState::$variant(ref mut hll) = $state {
            let arr = $array.as_any().downcast_ref::<$arrty>().unwrap();
            hll.extend(arr.iter().flatten().map(|s| s.to_string()));
        }
    }};
}
macro_rules! hll_extend_binary {
    ($state:expr, $array:expr, $variant:ident, $arrty:ty) => {{
        if let HLLState::$variant(ref mut hll) = $state {
            let arr = $array.as_any().downcast_ref::<$arrty>().unwrap();
            hll.extend(arr.iter().flatten().map(|b| b.to_vec()));
        }
    }};
}

impl HLLState {
    fn new(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::UInt8 => Some(Self::U8(HyperLogLog::new())),
            DataType::UInt16 => Some(Self::U16(HyperLogLog::new())),
            DataType::UInt32 => Some(Self::U32(HyperLogLog::new())),
            DataType::UInt64 => Some(Self::U64(HyperLogLog::new())),
            DataType::Int8 => Some(Self::I8(HyperLogLog::new())),
            DataType::Int16 => Some(Self::I16(HyperLogLog::new())),
            DataType::Int32 => Some(Self::I32(HyperLogLog::new())),
            DataType::Int64 => Some(Self::I64(HyperLogLog::new())),

            DataType::Utf8 => Some(Self::Str(HyperLogLog::new())),
            DataType::LargeUtf8 => Some(Self::LStr(HyperLogLog::new())),
            DataType::Utf8View => Some(Self::Str(HyperLogLog::new())),
            DataType::Binary => Some(Self::Bin(HyperLogLog::new())),
            DataType::LargeBinary => Some(Self::LBin(HyperLogLog::new())),
            _ => None,
        }
    }
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::U8(h) => h.to_bytes(),
            Self::U16(h) => h.to_bytes(),
            Self::U32(h) => h.to_bytes(),
            Self::U64(h) => h.to_bytes(),
            Self::I8(h) => h.to_bytes(),
            Self::I16(h) => h.to_bytes(),
            Self::I32(h) => h.to_bytes(),
            Self::I64(h) => h.to_bytes(),

            Self::Str(h) => h.to_bytes(),
            Self::LStr(h) => h.to_bytes(),
            Self::Bin(h) => h.to_bytes(),
            Self::LBin(h) => h.to_bytes(),
        }
    }
    // Use the HyperLogLog::count() method (with parentheses) to avoid resolving to
    // trait-based Iterator/StreamExt::count. This is evaluated only at finalization
    // to compute distinct counts from the accumulated hll_state.
    fn count(&self) -> u64 {
        match self {
            Self::U8(h) => h.count() as u64,
            Self::U16(h) => h.count() as u64,
            Self::U32(h) => h.count() as u64,
            Self::U64(h) => h.count() as u64,
            Self::I8(h) => h.count() as u64,
            Self::I16(h) => h.count() as u64,
            Self::I32(h) => h.count() as u64,
            Self::I64(h) => h.count() as u64,

            Self::Str(h) => h.count() as u64,
            Self::LStr(h) => h.count() as u64,
            Self::Bin(h) => h.count() as u64,
            Self::LBin(h) => h.count() as u64,
        }
    }
    fn extend(&mut self, dt: &DataType, array: &ArrayRef) {
        match dt {
            DataType::UInt8 => hll_extend_numeric!(self, array, U8, UInt8Array),
            DataType::UInt16 => hll_extend_numeric!(self, array, U16, UInt16Array),
            DataType::UInt32 => hll_extend_numeric!(self, array, U32, UInt32Array),
            DataType::UInt64 => hll_extend_numeric!(self, array, U64, UInt64Array),
            DataType::Int8 => hll_extend_numeric!(self, array, I8, Int8Array),
            DataType::Int16 => hll_extend_numeric!(self, array, I16, Int16Array),
            DataType::Int32 => hll_extend_numeric!(self, array, I32, Int32Array),
            DataType::Int64 => hll_extend_numeric!(self, array, I64, Int64Array),
            DataType::Float32 => {}
            DataType::Float64 => {}
            DataType::Utf8 => hll_extend_string!(self, array, Str, StringArray),
            DataType::LargeUtf8 => hll_extend_string!(self, array, LStr, LargeStringArray),
            DataType::Utf8View => hll_extend_string!(self, array, Str, StringViewArray),
            DataType::Binary => hll_extend_binary!(self, array, Bin, BinaryArray),
            DataType::LargeBinary => hll_extend_binary!(self, array, LBin, LargeBinaryArray),
            _ => {}
        }
    }
}

macro_rules! compute_avg_max_len {
    ($array:expr, $arrty:ty) => {{
        let arr = $array.as_any().downcast_ref::<$arrty>().unwrap();
        let mut total: u64 = 0;
        let mut max_len: u64 = 0;
        let mut count: u64 = 0;
        for v in arr.iter().flatten() {
            let l = v.len() as u64;
            total += l;
            if l > max_len {
                max_len = l;
            }
            count += 1;
        }
        Some((if count > 0 { total / count } else { 0 }, max_len))
    }};
}

/// Compute average and max length for variable-length arrays (strings/binary). Returns (avg_len, max_len).
fn avg_max_len(array: &ArrayRef) -> Option<(u64, u64)> {
    match array.data_type() {
        DataType::Utf8 => compute_avg_max_len!(array, StringArray),
        DataType::LargeUtf8 => compute_avg_max_len!(array, LargeStringArray),
        DataType::Binary => compute_avg_max_len!(array, BinaryArray),
        DataType::LargeBinary => compute_avg_max_len!(array, LargeBinaryArray),
        _ => None,
    }
}

/// Build a one-row RecordBatch for fragment stats given computed values.
fn build_fragment_stats_batch(
    field_id: u32,
    dt: &DataType,
    values: &ColumnStatistics,
) -> Result<RecordBatch> {
    let schema = Arc::new(ColumnStatistics::schema(dt));
    // Construct columns using ScalarValue for uniformity
    let cols: Vec<ScalarValue> = vec![
        ScalarValue::UInt32(Some(field_id)),
        values.min.clone().unwrap_or(ScalarValue::try_from(dt)?),
        values.max.clone().unwrap_or(ScalarValue::try_from(dt)?),
        ScalarValue::UInt64(Some(values.null_count.unwrap_or(0))),
        ScalarValue::UInt64(Some(values.nan_count.unwrap_or(0))),
        ScalarValue::UInt64(values.avg_len.map(Some).unwrap_or(None)),
        ScalarValue::UInt64(values.max_len.map(Some).unwrap_or(None)),
        ScalarValue::UInt64(values.distinct_count.map(Some).unwrap_or(None)),
        ScalarValue::Binary(values.hll_registers.clone()),
    ];
    let arrays = vec![
        Arc::new(UInt32Array::from(vec![field_id])) as ArrayRef,
        ScalarValue::iter_to_array(std::iter::once(cols[1].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[2].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(ScalarValue::UInt64(Some(
            values.null_count.unwrap_or(0),
        ))))?,
        ScalarValue::iter_to_array(std::iter::once(ScalarValue::UInt64(Some(
            values.nan_count.unwrap_or(0),
        ))))?,
        ScalarValue::iter_to_array(std::iter::once(cols[5].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[6].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[7].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[8].clone()))?,
    ];
    Ok(RecordBatch::try_new(schema, arrays)?)
}

/// Build a one-row RecordBatch for dataset stats given computed values.
fn build_dataset_stats_batch(
    field_id: u32,
    dt: &DataType,
    values: &ColumnStatistics,
) -> Result<RecordBatch> {
    let schema = Arc::new(ColumnStatistics::schema(dt));
    let cols: Vec<ScalarValue> = vec![
        ScalarValue::UInt32(Some(field_id)),
        values.min.clone().unwrap_or(ScalarValue::try_from(dt)?),
        values.max.clone().unwrap_or(ScalarValue::try_from(dt)?),
        ScalarValue::UInt64(Some(values.null_count.unwrap_or(0))),
        ScalarValue::UInt64(Some(values.nan_count.unwrap_or(0))),
        ScalarValue::UInt64(values.avg_len.map(Some).unwrap_or(None)),
        ScalarValue::UInt64(values.max_len.map(Some).unwrap_or(None)),
        ScalarValue::UInt64(values.distinct_count.map(Some).unwrap_or(None)),
    ];
    let arrays = vec![
        Arc::new(UInt32Array::from(vec![field_id])) as ArrayRef,
        ScalarValue::iter_to_array(std::iter::once(cols[1].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[2].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(ScalarValue::UInt64(Some(
            values.null_count.unwrap_or(0),
        ))))?,
        ScalarValue::iter_to_array(std::iter::once(ScalarValue::UInt64(Some(
            values.nan_count.unwrap_or(0),
        ))))?,
        ScalarValue::iter_to_array(std::iter::once(cols[5].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[6].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(cols[7].clone()))?,
        ScalarValue::iter_to_array(std::iter::once(ScalarValue::Binary(None)))?,
    ];
    Ok(RecordBatch::try_new(schema, arrays)?)
}

#[async_trait]
impl FragmentStatsExt for FileFragment {
    async fn analyze_column_statistics(&self, fields: Vec<Field>) -> Result<Vec<ColumnStatistics>> {
        let mut results = Vec::with_capacity(fields.len());
        let dataset = self.dataset();
        let frag_id = self.id() as u64;

        // Prepare: read existing stats and collect fields to scan in one pass
        let mut fields_to_scan: Vec<Field> = Vec::new();
        for f in fields.iter() {
            let field_id = f.id as u32;
            let path = fragment_colstats_path(dataset, frag_id, field_id);
            if dataset.object_store.exists(&path).await? {
                if let Some(rb) = read_fragment_colstats(dataset, frag_id, field_id).await? {
                    results.push(ColumnStatistics::from(field_id, &rb)?);
                    continue;
                }
            }
            // Need to compute stats for this field
            fields_to_scan.push(f.clone());
        }

        if fields_to_scan.is_empty() {
            return Ok(results);
        }

        // Build a projection schema including ALL requested fields (sorted by Field.id)
        let mut ids: Vec<i32> = fields_to_scan.iter().map(|f| f.id).collect();
        ids.sort_unstable();
        let projection: Schema = dataset.schema().project_by_ids(&ids, true);
        let reader = self.open(&projection, FragReadConfig::default()).await?;

        // Build field_id -> column_index mapping from projection schema (leaf-only)
        let mut id_to_index: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for (idx, f) in projection.fields_pre_order().enumerate() {
            if f.children.is_empty() {
                id_to_index.insert(f.id as u32, idx);
            }
        }

        // Initialize per-field accumulators and counters
        struct AccState {
            min_acc: MinAccumulator,
            max_acc: MaxAccumulator,
            null_count: u64,
            nan_count: u64,
            avg_len: Option<(u64, u64, u64)>, // (sum_len, max_len, count_non_null)
            hll_state: Option<HLLState>,
            dt: DataType,
        }
        let mut states: std::collections::HashMap<u32, AccState> = std::collections::HashMap::new();
        for f in fields_to_scan.iter() {
            let dt = f.data_type();
            states.insert(
                f.id as u32,
                AccState {
                    min_acc: MinAccumulator::try_new(&dt)?,
                    max_acc: MaxAccumulator::try_new(&dt)?,
                    null_count: 0,
                    nan_count: 0,
                    avg_len: None,
                    hll_state: HLLState::new(&dt),
                    dt,
                },
            );
        }

        // Iterate the stream of multi-column RecordBatches and update per-field stats
        let mut stream = reader.read_all(1024)?;
        use futures::StreamExt;
        while let Some(batch_fut) = stream.next().await {
            let batch = batch_fut.await?;
            if batch.num_columns() == 0 || batch.num_rows() == 0 {
                continue;
            }
            for f in fields_to_scan.iter() {
                let fid = f.id as u32;
                let Some(&col_idx) = id_to_index.get(&fid) else {
                    continue;
                };
                let array = batch.column(col_idx);

                if let Some(st) = states.get_mut(&fid) {
                    st.null_count += array.null_count() as u64;
                    st.nan_count += count_nans(array);
                    st.min_acc.update_batch(std::slice::from_ref(array))?;
                    st.max_acc.update_batch(std::slice::from_ref(array))?;

                    if let Some((avg_len, max_len)) = avg_max_len(array) {
                        let non_nulls = (batch.num_rows() - array.null_count()) as u64;
                        if let Some((sum, cur_max, cnt)) = st.avg_len.as_mut() {
                            *sum += avg_len * non_nulls;
                            *cur_max = (*cur_max).max(max_len);
                            *cnt += non_nulls;
                        } else {
                            st.avg_len = Some((avg_len * non_nulls, max_len, non_nulls));
                        }
                    }

                    if let Some(ref mut hs) = st.hll_state {
                        hs.extend(&st.dt, array);
                    }
                }
            }
        }

        // Finalize: construct and persist ColumnStatsFragmentResult per field
        for f in fields_to_scan.into_iter() {
            let field_id = f.id as u32;
            if let Some(mut st) = states.remove(&field_id) {
                let min = st.min_acc.evaluate().ok();
                let max = st.max_acc.evaluate().ok();
                let (avg_len, max_len) = if let Some((sum, max_l, cnt)) = st.avg_len {
                    let avg = if cnt > 0 { sum / cnt } else { 0 };
                    (Some(avg), Some(max_l))
                } else {
                    (None, None)
                };

                // Finalize HLL once at the end per field: compute registers and distinct directly
                let (distinct_count, hll_registers) = st
                    .hll_state
                    .as_ref()
                    .map(|hs| (Some(hs.count()), Some(hs.to_bytes())))
                    .unwrap_or((None, None));

                let result = ColumnStatistics {
                    field_id,
                    min,
                    max,
                    null_count: Some(st.null_count),
                    nan_count: Some(st.nan_count),
                    distinct_count,
                    avg_len,
                    max_len,
                    hll_registers,
                };

                // Persist fragment-level stats as one-row batch
                let rb = build_fragment_stats_batch(field_id, &st.dt, &result)?;
                write_fragment_colstats(dataset, frag_id, field_id, rb).await?;
                results.push(result);
            }
        }

        Ok(results)
    }
}

#[async_trait]
impl DatasetStatsExt for Dataset {
    async fn analyze_column_statistics(&self) -> Result<Vec<ColumnStatistics>> {
        // Select fields enabled via metadata and leaf-only
        let enabled_fields: Vec<Field> = self
            .schema()
            .fields_pre_order()
            .filter(|f| f.children.is_empty())
            .filter(|f| {
                f.metadata
                    .get("lance.stats.enabled")
                    .map(|v| v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if enabled_fields.is_empty() {
            return Ok(Vec::new());
        }

        // Gather fragment-level stats
        let mut per_field: std::collections::HashMap<u32, ColumnStatistics> =
            std::collections::HashMap::new();
        let mut hll_merge: std::collections::HashMap<u32, HyperLogLog<u64>> =
            std::collections::HashMap::new();

        for frag in self.get_fragments() {
            let frag_results = frag
                .analyze_column_statistics(enabled_fields.clone())
                .await?;
            for fr in frag_results {
                let entry = per_field.entry(fr.field_id).or_insert(ColumnStatistics {
                    field_id: fr.field_id,
                    min: None,
                    max: None,
                    null_count: Some(0),
                    nan_count: Some(0),
                    distinct_count: None,
                    avg_len: None,
                    max_len: None,
                    hll_registers: None,
                });

                // Merge min/max without moving existing values
                if let Some(m) = fr.min.clone() {
                    match entry.min.as_ref() {
                        None => {
                            entry.min = Some(m);
                        }
                        Some(a) => {
                            if let Some(ord) = a.partial_cmp(&m) {
                                if ord == std::cmp::Ordering::Greater {
                                    entry.min = Some(m);
                                }
                            }
                        }
                    }
                }
                if let Some(x) = fr.max.clone() {
                    match entry.max.as_ref() {
                        None => {
                            entry.max = Some(x);
                        }
                        Some(a) => {
                            if let Some(ord) = a.partial_cmp(&x) {
                                if ord == std::cmp::Ordering::Less {
                                    entry.max = Some(x);
                                }
                            }
                        }
                    }
                }
                // Sum counts
                if let Some(nc) = fr.null_count {
                    entry.null_count = Some(entry.null_count.unwrap_or(0) + nc);
                }
                if let Some(nn) = fr.nan_count {
                    entry.nan_count = Some(entry.nan_count.unwrap_or(0) + nn);
                }
                // Max of max_len (approx)
                if let Some(ml) = fr.max_len {
                    entry.max_len = Some(entry.max_len.unwrap_or(0).max(ml));
                }
                // Keep avg_len as None (requires weighted merge); could set simple average if desired

                // Merge HLL
                if let Some(regs) = fr.hll_registers.as_ref() {
                    let other = HyperLogLog::<u64>::from_bytes(regs)?;
                    hll_merge
                        .entry(fr.field_id)
                        .and_modify(|base| base.merge(&other))
                        .or_insert(other);
                }
            }
        }

        // Finalize distinct_count and persist dataset-level stats per field
        let mut results = Vec::with_capacity(per_field.len());
        for field in enabled_fields.iter() {
            let field_id = field.id as u32;
            if let Some(mut ds) = per_field.remove(&field_id) {
                if let Some(hll) = hll_merge.remove(&field_id) {
                    ds.distinct_count = Some(hll.count() as u64);
                }
                // Persist one-row stats file
                let rb = build_dataset_stats_batch(field_id, &field.data_type(), &ds)?;
                write_dataset_colstats(self, self.manifest.version, field_id, rb).await?;
                results.push(ds);
            }
        }

        Ok(results)
    }

    async fn column_statistics(&self) -> Result<Vec<ColumnStatistics>> {
        // Attempt to read stats files for current manifest version and enabled fields
        let enabled_fields: Vec<Field> = self
            .schema()
            .fields_pre_order()
            .filter(|f| f.children.is_empty())
            .filter(|f| {
                f.metadata
                    .get("lance.stats.enabled")
                    .map(|v| v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let mut out = Vec::new();
        for f in enabled_fields.iter() {
            let field_id = f.id as u32;
            if let Some(rb) = read_dataset_colstats(self, self.manifest.version, field_id).await? {
                out.push(ColumnStatistics::from(field_id, &rb)?);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    // SPDX-License-Identifier: Apache-2.0
    // SPDX-FileCopyrightText: Copyright The Lance Authors

    //! Tests for dataset-level column statistics persisted under `_stats`.
    //!
    //! These tests use the in-memory object store (memory://) per project guidance.

    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::dataset::stats::io::{dataset_colstats_path, fragment_colstats_path};
    use crate::dataset::stats::{ColumnStatistics, DatasetStatsExt, FragmentStatsExt};
    use crate::dataset::write::{InsertBuilder, WriteMode, WriteParams};
    use crate::dataset::Dataset;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::Result;

    /// Build an Arrow schema with optional per-field metadata.
    fn make_schema(fields: Vec<(String, DataType, bool, bool)>) -> ArrowSchema {
        // tuple: (name, dtype, nullable, stats_enabled)
        let mut arrow_fields = Vec::with_capacity(fields.len());
        for (name, dt, nullable, stats_enabled) in fields.into_iter() {
            let mut f = ArrowField::new(name, dt, nullable);
            if stats_enabled {
                let mut md = HashMap::new();
                md.insert("lance.stats.enabled".to_string(), "true".to_string());
                f = f.with_metadata(md);
            }
            arrow_fields.push(f);
        }
        ArrowSchema::new(arrow_fields)
    }

    /// Helper: create a dataset at given URI with a single RecordBatch.
    async fn create_dataset_with_batch(uri: &str, batch: RecordBatch) -> Result<Arc<Dataset>> {
        let ds = InsertBuilder::new(uri)
            .with_params(&WriteParams {
                mode: WriteMode::Create,
                ..Default::default()
            })
            .execute(vec![batch])
            .await?;
        Ok(Arc::new(ds))
    }

    /// Helper: append data (creates a new fragment).
    async fn append_batch(ds: Arc<Dataset>, batch: RecordBatch) -> Result<Arc<Dataset>> {
        let ds2 = InsertBuilder::new(ds.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            })
            .execute(vec![batch])
            .await?;
        Ok(Arc::new(ds2))
    }

    /// Retrieve field_id by name
    fn field_id(ds: &Dataset, name: &str) -> u32 {
        ds.schema()
            .fields_pre_order()
            .find(|f| f.name == name)
            .map(|f| f.id as u32)
            .expect("field must exist")
    }

    /// Retrieve first fragment id (for single-fragment tests)
    fn first_frag_id(ds: &Dataset) -> u64 {
        ds.get_fragments()[0].id() as u64
    }

    /// Assert a ColumnStatsDatasetResult for integer min/max/null/distinct.
    fn assert_int_stats(
        stats: &ColumnStatistics,
        expected_min: i32,
        expected_max: i32,
        expected_nulls: u64,
        expected_distinct_at_least: u64,
    ) {
        use datafusion::scalar::ScalarValue;
        assert_eq!(stats.min, Some(ScalarValue::Int32(Some(expected_min))));
        assert_eq!(stats.max, Some(ScalarValue::Int32(Some(expected_max))));
        assert_eq!(stats.null_count, Some(expected_nulls));
        if let Some(dc) = stats.distinct_count {
            assert!(
                dc >= expected_distinct_at_least,
                "distinct_count {} < {}",
                dc,
                expected_distinct_at_least
            );
        }
    }

    #[tokio::test]
    async fn test_no_fields_enabled() -> Result<()> {
        // Dataset with no lance.stats.enabled metadata
        let schema = make_schema(vec![
            ("id".to_string(), DataType::Int32, true, false),
            ("name".to_string(), DataType::Utf8, true, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as _,
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])) as _,
            ],
        )?;

        let ds = create_dataset_with_batch("memory://", batch).await?;

        // Analyze: expect empty
        let analyzed = ds.analyze_column_statistics().await?;
        assert!(analyzed.is_empty());

        // Read existing stats: expect empty
        let existing = ds.column_statistics().await?;
        assert!(existing.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_one_field_enabled() -> Result<()> {
        // Dataset with one enabled int field `val`
        let schema = make_schema(vec![("val".to_string(), DataType::Int32, true, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                None,
            ])) as _],
        )?;
        let ds = create_dataset_with_batch("memory://", batch).await?;

        // Run analysis
        let results = ds.analyze_column_statistics().await?;
        assert_eq!(results.len(), 1);
        let stats = &results[0];
        assert_eq!(stats.field_id, field_id(&ds, "val"));
        assert_int_stats(stats, 1, 3, 1, 3);

        // Verify stats files exist
        let fid = field_id(&ds, "val");
        let frag_id = first_frag_id(&ds);
        let frag_path = fragment_colstats_path(&ds, frag_id, fid);
        assert!(ds.object_store.exists(&frag_path).await?);
        let ds_path = dataset_colstats_path(&ds, ds.manifest.version, fid);
        assert!(ds.object_store.exists(&ds_path).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_fields_enabled() -> Result<()> {
        // Enable both val:Int32 and name:Utf8
        let schema = make_schema(vec![
            ("val".to_string(), DataType::Int32, true, true),
            ("name".to_string(), DataType::Utf8, true, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(1)])) as _,
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("bbb"),
                    Some(""),
                    None,
                ])) as _,
            ],
        )?;
        let ds = create_dataset_with_batch("memory://", batch).await?;

        let results = ds.analyze_column_statistics().await?;
        assert_eq!(results.len(), 2);
        // Find stats by field_id
        let val_id = field_id(&ds, "val");
        let name_id = field_id(&ds, "name");
        let val_stats = results
            .iter()
            .find(|s| s.field_id == val_id)
            .expect("val stats");
        let name_stats = results
            .iter()
            .find(|s| s.field_id == name_id)
            .expect("name stats");

        assert_int_stats(val_stats, 1, 2, 1, 2);

        // String stats: max_len should be present (bbb -> 3), avg_len may be None (dataset-level), allow None
        assert!(
            name_stats.max_len.is_some(),
            "string max_len should be present at dataset-level"
        );
        // avg_len may be None by design at dataset-level; allow None or Some

        // Ensure dataset-level files exist for both fields
        let ds_ver = ds.manifest.version;
        let path_val = dataset_colstats_path(&ds, ds_ver, val_id);
        let path_name = dataset_colstats_path(&ds, ds_ver, name_id);
        assert!(ds.object_store.exists(&path_val).await?);
        assert!(ds.object_store.exists(&path_name).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_single_fragment_dataset_stats() -> Result<()> {
        // One fragment with two enabled fields
        let schema = make_schema(vec![
            ("val".to_string(), DataType::Int32, true, true),
            ("name".to_string(), DataType::Utf8, true, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(2), None])) as _,
                Arc::new(StringArray::from(vec![
                    Some("aa"),
                    Some("b"),
                    None,
                    Some("cccc"),
                ])) as _,
            ],
        )?;
        let ds = create_dataset_with_batch("memory://", batch).await?;

        // Fragment-level stats
        let fields: Vec<_> = ds
            .schema()
            .fields_pre_order()
            .filter(|f| {
                f.metadata
                    .get("lance.stats.enabled")
                    .map(|v| v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        assert!(!fields.is_empty());
        let frag = ds.get_fragments()[0].clone();
        let frag_stats = frag.analyze_column_statistics(fields).await?;

        // Dataset-level stats
        let ds_stats = ds.analyze_column_statistics().await?;

        // Compare aggregation: for val
        let val_id = field_id(&ds, "val");
        let f_val = frag_stats.iter().find(|s| s.field_id == val_id).unwrap();
        let d_val = ds_stats.iter().find(|s| s.field_id == val_id).unwrap();
        // Min/Max should match
        assert_eq!(f_val.min, d_val.min);
        assert_eq!(f_val.max, d_val.max);
        // Nulls summed (single fragment -> equal)
        assert_eq!(f_val.null_count, d_val.null_count);
        // Distinct: dataset should be >= fragment (equal for single fragment)
        if let (Some(df), Some(ff)) = (d_val.distinct_count, f_val.distinct_count) {
            assert!(df >= ff);
        }

        // Verify dataset-level files exist
        let ds_ver = ds.manifest.version;
        let path_val = dataset_colstats_path(&ds, ds_ver, val_id);
        assert!(ds.object_store.exists(&path_val).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_fragment_dataset_stats() -> Result<()> {
        // Enabled fields
        let schema = make_schema(vec![
            ("val".to_string(), DataType::Int32, true, true),
            ("name".to_string(), DataType::Utf8, true, true),
        ]);
        // First fragment
        let batch1 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(3)])) as _,
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("bb"),
                    None,
                    Some("ccc"),
                ])) as _,
            ],
        )?;
        let ds = create_dataset_with_batch("memory://", batch1).await?;

        // Second fragment append
        let batch2 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![Some(3), Some(4), Some(2), None])) as _,
                Arc::new(StringArray::from(vec![
                    Some("dd"),
                    None,
                    Some("bb"),
                    Some("eeee"),
                ])) as _,
            ],
        )?;
        let ds2 = append_batch(ds.clone(), batch2).await?;

        // Analyze merged stats
        let ds_stats = ds2.analyze_column_statistics().await?;
        let val_id = field_id(&ds2, "val");
        let name_id = field_id(&ds2, "name");
        let val_stats = ds_stats.iter().find(|s| s.field_id == val_id).unwrap();
        let name_stats = ds_stats.iter().find(|s| s.field_id == name_id).unwrap();

        // Int column: min/max across both fragments, null_count summed, distinct_count reflect unique {1,2,3,4} => 4
        assert_int_stats(val_stats, 1, 4, 2, 4);

        // String column: max_len reflect across fragments ("eeee" -> 4), avg_len may be None; allow None
        assert_eq!(name_stats.max_len, Some(4));

        // Verify fragment-level files exist for both fragments and the int field
        let fid = val_id;
        for frag in ds2.get_fragments() {
            let fp = fragment_colstats_path(&ds2, frag.id() as u64, fid);
            assert!(ds2.object_store.exists(&fp).await?);
        }
        // Verify dataset-level files exist for latest version
        let ds_ver = ds2.manifest.version;
        let path_val = dataset_colstats_path(&ds2, ds_ver, val_id);
        let path_name = dataset_colstats_path(&ds2, ds_ver, name_id);
        assert!(ds2.object_store.exists(&path_val).await?);
        assert!(ds2.object_store.exists(&path_name).await?);

        Ok(())
    }
}
