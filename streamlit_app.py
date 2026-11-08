"""Entry point for Streamlit Community Cloud (its default main file name). Runs app.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
