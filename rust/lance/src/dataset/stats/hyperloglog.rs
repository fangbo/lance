// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Placeholder for HyperLogLog implementation used in column statistics.
//! A copy of DataFusion's HyperLogLog will be adapted here with serialization
//! and deserialization helpers for fragment-level merging.

use std::hash::Hash;
use std::marker::PhantomData;

use ahash::RandomState;
use lance_core::{Error, Result};

/// The greater is P, the smaller the error.
const HLL_P: usize = 14_usize;
/// The number of bits of the hash value used determining the number of leading zeros
const HLL_Q: usize = 64_usize - HLL_P;
const NUM_REGISTERS: usize = 1_usize << HLL_P;
/// Mask to obtain index into the registers
const HLL_P_MASK: u64 = (NUM_REGISTERS as u64) - 1;

/// Fixed seed for the hashing so that values are consistent across runs
const SEED: RandomState = RandomState::with_seeds(
    0x885f6cab121d01a3_u64,
    0x71e4379f2976ad8f_u64,
    0xbf30173dd28a8816_u64,
    0x0eaea5d736d733a4_u64,
);

/// A compact HyperLogLog implementation copied from DataFusion with minor adjustments.
#[derive(Clone, Debug)]
pub struct HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    registers: [u8; NUM_REGISTERS],
    phantom: PhantomData<T>,
}

impl<T> Default for HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HyperLogLog<T>
where
    T: Hash + ?Sized,
{
    /// Creates a new, empty HyperLogLog.
    pub fn new() -> Self {
        let registers = [0; NUM_REGISTERS];
        Self::new_with_registers(registers)
    }

    /// Creates a HyperLogLog from already populated registers.
    pub fn new_with_registers(registers: [u8; NUM_REGISTERS]) -> Self {
        Self {
            registers,
            phantom: PhantomData,
        }
    }

    #[inline]
    fn hash_value(&self, obj: &T) -> u64 {
        SEED.hash_one(obj)
    }

    /// Adds an element to the HyperLogLog.
    pub fn add(&mut self, obj: &T) {
        let hash = self.hash_value(obj);
        let index = (hash & HLL_P_MASK) as usize;
        let p = ((hash >> HLL_P) | (1_u64 << HLL_Q)).trailing_zeros() + 1;
        self.registers[index] = self.registers[index].max(p as u8);
    }

    #[inline]
    fn get_histogram(&self) -> [u32; HLL_Q + 2] {
        let mut histogram = [0; HLL_Q + 2];
        for r in self.registers {
            histogram[r as usize] += 1;
        }
        histogram
    }

    /// Merge the other [`HyperLogLog`] into this one.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.registers.len(), other.registers.len());
        for i in 0..self.registers.len() {
            self.registers[i] = self.registers[i].max(other.registers[i]);
        }
    }

    /// Guess the number of unique elements seen by the HyperLogLog.
    pub fn count(&self) -> usize {
        let histogram = self.get_histogram();
        let m = NUM_REGISTERS as f64;
        let mut z = m * hll_tau((m - histogram[HLL_Q + 1] as f64) / m);
        for i in histogram[1..=HLL_Q].iter().rev() {
            z += *i as f64;
            z *= 0.5;
        }
        z += m * hll_sigma(histogram[0] as f64 / m);
        (0.5 / 2_f64.ln() * m * m / z).round() as usize
    }

    /// Serialize registers into bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.registers.to_vec()
    }

    /// Deserialize registers from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let arr: [u8; NUM_REGISTERS] = bytes.try_into().map_err(|_| Error::InvalidInput {
            source: "Invalid HLL register bytes".into(),
            location: snafu::location!(),
        })?;
        Ok(Self::new_with_registers(arr))
    }
}

/// Helper function sigma as defined in
/// "New cardinality estimation algorithms for HyperLogLog sketches"
/// Otmar Ertl, arXiv:1702.01284
#[inline]
fn hll_sigma(x: f64) -> f64 {
    if x == 1. {
        f64::INFINITY
    } else {
        let mut y = 1.0;
        let mut z = x;
        let mut x = x;
        loop {
            x *= x;
            let z_prime = z;
            z += x * y;
            y += y;
            if z_prime == z {
                break;
            }
        }
        z
    }
}

/// Helper function tau as defined in
/// "New cardinality estimation algorithms for HyperLogLog sketches"
/// Otmar Ertl, arXiv:1702.01284
#[inline]
fn hll_tau(x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        0.0
    } else {
        let mut y = 1.0;
        let mut z = 1.0 - x;
        let mut x = x;
        loop {
            x = x.sqrt();
            let z_prime = z;
            y *= 0.5;
            z -= (1.0 - x).powi(2) * y;
            if z_prime == z {
                break;
            }
        }
        z / 3.0
    }
}

impl<T> Extend<T> for HyperLogLog<T>
where
    T: Hash,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(&elem);
        }
    }
}

impl<'a, T> Extend<&'a T> for HyperLogLog<T>
where
    T: 'a + Hash + ?Sized,
{
    fn extend<S: IntoIterator<Item = &'a T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(elem);
        }
    }
}
