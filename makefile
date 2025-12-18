# Makefile
.PHONY: dev docs test 

dev:
\tpip install -U pip maturin
\tmaturin develop --release -m ts2net-rs/Cargo.toml
\tpip install -e .[speed,dtw,approx,rdata]

test:
\tpytest -q

docs:
\tpip install mkdocs-material
\tmkdocs serve -a 0.0.0.0:8000
