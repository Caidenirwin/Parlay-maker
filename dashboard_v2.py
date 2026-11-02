import os
import runpy
import sys
import streamlit as st

# Parlay Quant 2.0 (black + green) is now the main dashboard - this file just opens it, so both of these work:
#   python -m streamlit run dashboard.py
#   python dashboard_v2.py
# The old blue look:  python dashboard_classic.py

if __name__ == "__main__" and not st.runtime.exists():
    # started with plain "python dashboard_v2.py" -> start Streamlit ourselves
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    sys.exit(cli.main())

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py"), run_name="__main__")
