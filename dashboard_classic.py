import os
import runpy
import sys
import streamlit as st

# The ORIGINAL blue look of Parlay Quant, before 2.0 - same dashboard (it runs dashboard.py), old colours.
# Run it with:   python dashboard_classic.py

# the original theme (2.0's black theme lives in .streamlit/config.toml)
CLASSIC = {"primaryColor": "#3B82F6", "backgroundColor": "#0B0F17", "secondaryBackgroundColor": "#121826",
           "textColor": "#E5E7EB", "borderColor": "#1F2937", "dataframeHeaderBackgroundColor": "#121826",
           "baseRadius": "10px"}

if __name__ == "__main__" and not st.runtime.exists():
    # started with plain "python dashboard_classic.py" -> start Streamlit ourselves, with the old colours
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + \
               [arg for key, value in CLASSIC.items() for arg in (f"--theme.{key}", value)] + sys.argv[1:]
    sys.exit(cli.main())

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py"), run_name="__main__",
               init_globals={"PQ_THEME": "classic"})   # tells dashboard.py to skip the 2.0 styles
