//! ts2net-rs - High-performance time series to network conversion in Rust
//!
//! This crate provides efficient implementations of various time series to
//! network conversion algorithms, including visibility graphs, recurrence
//! networks, and more.
//!
//! # Using it from Rust
//!
//! The algorithm modules are plain Rust over [`ndarray`] and carry no Python
//! dependency. Turn the bindings off to get a PyO3-free build:
//!
//! ```toml
//! [dependencies]
//! ts2net_rs = { version = "0.9", default-features = false }
//! ```
//!
//! ```no_run
//! use ndarray::array;
//! use ts2net_rs::graphs::{hvg_edges, clustering_avg};
//!
//! let y = array![1.0, 2.0, 1.0, 2.0];
//! let edges = hvg_edges(&y);
//! let pairs: Vec<(usize, usize)> = (0..edges.nrows())
//!     .map(|r| (edges[[r, 0]] as usize, edges[[r, 1]] as usize))
//!     .collect();
//! let c = clustering_avg(y.len(), &pairs);
//! ```
//!
//! # Using it from Python
//!
//! The default `python` feature compiles the [`mod@python`] module into a
//! CPython extension named `ts2net_rs`, built by maturin. Every function it
//! exposes is a thin adapter over the modules below.

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod distance;
pub mod embedding;
pub mod graphs;
pub mod sindy;
pub mod utils;

#[cfg(feature = "python")]
pub mod python;
