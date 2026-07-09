"""Root conftest — ensures the repo root is importable (`from backend import ...`)
no matter how pytest is invoked."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
