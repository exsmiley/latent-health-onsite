"""`python -m rag.evals [options]` is the same as `rag eval [options]`."""

import sys

from rag.cli import app

if __name__ == "__main__":
    app(["eval", *sys.argv[1:]], prog_name="rag")
