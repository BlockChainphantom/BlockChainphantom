"""Entry point so the package can be run as ``python -m phantom_privacy``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
