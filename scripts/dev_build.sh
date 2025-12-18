# scripts/dev_build.sh
set -euo pipefail
pip install -U pip maturin
maturin develop --release -m ts2net-rs/Cargo.toml
pip install -e .[speed,dtw,approx,rdata]
pytest -q
