"""Numerical prior constants, independent of TensorFlow record serialization."""
from datetime import date

BASE_START = date.fromisoformat("1885-01-01").toordinal()
BASE_END = date.fromisoformat("2023-12-31").toordinal() + 1
CONTEXT_LENGTH = 1_000
