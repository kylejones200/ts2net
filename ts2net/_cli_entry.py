"""Console script entry point with optional pipeline dependencies."""


def main() -> None:
    try:
        from ts2net.cli import main as _cli_main
    except ImportError as exc:
        raise SystemExit(
            "ts2net CLI requires optional dependencies. "
            "Install with: pip install ts2net[pipeline]"
        ) from exc
    _cli_main()
