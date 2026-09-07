//! Architectural gate: the Rust-only build must not depend on Python.
//!
//! `default-features = false` is how another Rust crate consumes ts2net_rs.
//! That configuration must contain no PyO3 and no NumPy, or the crate is a
//! Python extension with an `rlib` label on it rather than a real library.
//!
//! This is a contract, so it is asserted here rather than checked by hand.

use std::collections::BTreeSet;
use std::process::Command;

/// Crates that must never appear in a `default-features = false` build.
const FORBIDDEN: &[&str] = &[
    "pyo3",
    "pyo3-build-config",
    "pyo3-ffi",
    "pyo3-macros",
    "pyo3-macros-backend",
    "numpy",
];

/// Crate names in the normal (non-dev, non-build) dependency graph.
fn dependency_crate_names(extra_args: &[&str]) -> BTreeSet<String> {
    let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let mut args = vec![
        "tree",
        "--manifest-path",
        manifest,
        // Normal edges only: a dev-dependency is not something a consumer
        // links, so it is not part of this contract.
        "--edges",
        "normal",
        "--prefix",
        "none",
        "--no-dedupe",
    ];
    args.extend_from_slice(extra_args);

    let out = Command::new(env!("CARGO"))
        .args(&args)
        .output()
        .expect("failed to run `cargo tree`");
    assert!(
        out.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn rust_only_build_contains_no_python_dependencies() {
    let crates = dependency_crate_names(&["--no-default-features"]);

    // Guard against a silently empty parse making this vacuously true.
    assert!(
        crates.contains("ts2net_rs"),
        "cargo tree output did not name the crate itself; parse is wrong: {crates:?}"
    );
    assert!(
        crates.contains("ndarray"),
        "expected ndarray in the Rust-only build: {crates:?}"
    );

    let found: Vec<&str> = FORBIDDEN
        .iter()
        .copied()
        .filter(|c| crates.contains(*c))
        .collect();
    assert!(
        found.is_empty(),
        "`default-features = false` must be free of Python dependencies, found {found:?}.\n\
         Something moved a pyo3/numpy-using item outside `#[cfg(feature = \"python\")]`, \
         or made one of those crates a non-optional dependency."
    );
}

#[test]
fn the_python_build_does_contain_them() {
    // The mirror image: proves the gate above is discriminating rather than
    // passing because the tree is empty or the names are spelled wrong.
    let crates = dependency_crate_names(&[]);
    assert!(crates.contains("pyo3"), "expected pyo3 in the default build");
    assert!(crates.contains("numpy"), "expected numpy in the default build");
}
