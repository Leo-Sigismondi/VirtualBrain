#!/usr/bin/env python3
"""
Preprocessing wrapper script
Run this to preprocess BCI Competition IV 2a dataset
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import main

if __name__ == "__main__":
    main()
