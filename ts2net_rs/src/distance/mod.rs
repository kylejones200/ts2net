//! Distance, similarity and neighbour-search primitives.
//!
//! Every function here is pure Rust operating on [`ndarray`] types; the PyO3
//! wrappers in the crate root are thin adapters over these.

mod dtw;
mod event_sync;
mod kdtree;
mod metrics;

pub use dtw::{cdist_dtw, cdist_dtw_rectangular, dtw_pair};
pub use event_sync::{event_sync, EventSync};
pub use kdtree::{knn, radius, MAX_KDTREE_DIM};
pub use metrics::{pair_dist, pearson, Metric};
