"""Prepared by 2025ag05290"""

"""Dependency-ordered scheduled Python entry point; exits nonzero on failure."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import run
if __name__ == '__main__': run()
