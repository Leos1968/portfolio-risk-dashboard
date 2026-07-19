"""Multipage entrypoint — Market Command Center + IB/PE Modeling Suite.

This is the file Streamlit Cloud deploys (entrypoint ``analyzer/dashboard.py``).
Mounting the modeling suite as a second page here means it ships on the *existing*
deployment on push-to-main — no separate Streamlit Cloud app to configure.

The two pages live in ``market_command_center.py`` (the original dashboard) and
``ibpe_modeling.py`` (which renders ``modeling.ui``).
"""

import streamlit as st

st.set_page_config(page_title="Markets & Modeling — Jeriel De Leon", page_icon="🛰️", layout="wide")

pages = [
    st.Page("market_command_center.py", title="Market Command Center", icon="🛰️", default=True),
    st.Page("ibpe_modeling.py", title="IB/PE Modeling Suite", icon="📐"),
]
st.navigation(pages).run()
