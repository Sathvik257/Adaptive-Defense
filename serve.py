"""Portable hosting entry point. Honors the platform-provided PORT."""
import os
import sys
from pathlib import Path
from streamlit.web import cli

if __name__ == "__main__":
    port=int(os.environ.get("PORT","8501"))
    if not 1<=port<=65535:
        raise ValueError("PORT must be between 1 and 65535")
    os.chdir(Path(__file__).resolve().parent)
    sys.argv=["streamlit","run","dashboard.py","--server.address=0.0.0.0",
              f"--server.port={port}","--server.headless=true"]
    sys.exit(cli.main())
