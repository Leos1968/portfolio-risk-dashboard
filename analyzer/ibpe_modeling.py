"""IB/PE Modeling Suite — a page of the multipage app (see dashboard.py).

Runs under st.navigation, which re-executes this file on every interaction, so
the whole interface is drawn by modeling.ui.render(). Page config is owned by
the entrypoint, so this page does not set it.
"""

import os
import sys

# Repo root on the path so the `modeling` package resolves from analyzer/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.ui import render  # noqa: E402

render()
