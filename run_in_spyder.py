"""Open this file in Spyder and press F5. No browser or extra pip packages."""
from pathlib import Path
import sys

# Resolve imports from this file, regardless of Spyder's working directory.
PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from simulate import main

if __name__ == "__main__":
    # Explicit arguments avoid inheriting IPython/Spyder command-line options.
    main([])
