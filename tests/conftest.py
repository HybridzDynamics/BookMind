"""
Pytest configuration and environment fixtures for BookMind test suite.

Ensures the project root directory is present in sys.path during test execution.
Note: Strictly functional implementation using only `def` (no `class`).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
