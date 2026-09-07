//! Surrogate generation and significance testing.

mod stats;
mod surrogates;

pub use stats::{corr_perm, moran_i};
pub use surrogates::{iaaft, iaaft_legacy, surrogate_phase};
