"""Allow running the decision module as: python -m src.betting.decision"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
