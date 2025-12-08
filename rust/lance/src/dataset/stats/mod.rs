// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset-level column statistics scaffolding.
//! This module mirrors the DatasetIndexExt approach and prepares traits and storage helpers
//! for fragment-level and dataset-level column statistics persisted under the `_stats` directory.

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use async_trait::async_trait;
use datafusion::scalar::ScalarValue;
use lance_core::datatypes::Field;
use lance_core::Result;

pub mod colstats;
pub(crate) mod hyperloglog;
pub(crate) mod io;

pub struct ColumnStatistics {
    pub field_id: u32,
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
    pub null_count: Option<u64>,
    pub nan_count: Option<u64>,
    pub distinct_count: Option<u64>,
    pub avg_len: Option<u64>,
    pub max_len: Option<u64>,
    pub hll_registers: Option<Vec<u8>>, // serialized HLL registers
}

impl ColumnStatistics {
    pub(crate) fn schema(field_arrow_type: &DataType) -> ArrowSchema {
        ArrowSchema::new(vec![
            ArrowField::new("field_id", DataType::UInt32, false),
            ArrowField::new("min", field_arrow_type.clone(), true),
            ArrowField::new("max", field_arrow_type.clone(), true),
            ArrowField::new("null_count", DataType::UInt64, false),
            ArrowField::new("nan_count", DataType::UInt64, false),
            ArrowField::new("avg_len", DataType::UInt64, true),
            ArrowField::new("max_len", DataType::UInt64, true),
            ArrowField::new("distinct_count", DataType::UInt64, true),
            ArrowField::new("hll_registers", DataType::Binary, true),
        ])
    }

    pub(crate) fn from(field_id: u32, rb: &RecordBatch) -> Result<Self> {
        let min = rb
            .column_by_name("min")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok());
        let max = rb
            .column_by_name("max")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok());
        let null_count = rb
            .column_by_name("null_count")
            .and_then(|arr| arr.as_any().downcast_ref::<arrow_array::UInt64Array>())
            .map(|a| a.value(0));
        let nan_count = rb
            .column_by_name("nan_count")
            .and_then(|arr| arr.as_any().downcast_ref::<arrow_array::UInt64Array>())
            .map(|a| a.value(0));
        let avg_len = rb
            .column_by_name("avg_len")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok())
            .and_then(|sv| match sv {
                ScalarValue::UInt64(v) => v,
                _ => None,
            });
        let max_len = rb
            .column_by_name("max_len")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok())
            .and_then(|sv| match sv {
                ScalarValue::UInt64(v) => v,
                _ => None,
            });
        let distinct_count = rb
            .column_by_name("distinct_count")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok())
            .and_then(|sv| match sv {
                ScalarValue::UInt64(v) => v,
                _ => None,
            });
        let hll_registers = rb
            .column_by_name("hll_registers")
            .and_then(|arr| ScalarValue::try_from_array(arr, 0).ok())
            .and_then(|sv| match sv {
                ScalarValue::Binary(v) => v,
                _ => None,
            });
        Ok(Self {
            field_id,
            min,
            max,
            null_count,
            nan_count,
            distinct_count,
            avg_len,
            max_len,
            hll_registers,
        })
    }
}

#[async_trait]
pub trait FragmentStatsExt {
    async fn analyze_column_statistics(&self, fields: Vec<Field>) -> Result<Vec<ColumnStatistics>>;
}

#[async_trait]
pub trait DatasetStatsExt {
    async fn analyze_column_statistics(&self) -> Result<Vec<ColumnStatistics>>;
    async fn column_statistics(&self) -> Result<Vec<ColumnStatistics>>;
}
