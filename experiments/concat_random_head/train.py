"""Portable four-GPU training entry point; see the repository README."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.launch import main
if __name__ == '__main__':
    main(Path(__file__).resolve().parent.name)
