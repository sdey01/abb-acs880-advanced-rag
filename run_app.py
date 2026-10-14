"""Launch Streamlit after preloading CPU PyTorch on Windows."""
from __future__ import annotations

import sys
from pathlib import Path

import torch  # Load native CPU runtime before Streamlit initializes its numerical stack.

from config import ROOT


def main() -> None:
    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), *sys.argv[1:]]
    streamlit_main()


if __name__ == "__main__":
    main()
