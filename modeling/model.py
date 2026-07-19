"""Standalone entrypoint for the IB/PE Modeling Suite.

Run with:  streamlit run modeling/model.py

The same UI is also mounted as a page inside the deployed Market Command Center
(see analyzer/ibpe_modeling.py); both call modeling.ui.render(), so there is a
single source of truth for the interface.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Put the repo root on the path so `import modeling.*` resolves when Streamlit
# runs this file directly (its own directory, not the repo root, is on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.ui import render  # noqa: E402

st.set_page_config(page_title="IB/PE Modeling Suite", page_icon="📐", layout="wide")
render()
