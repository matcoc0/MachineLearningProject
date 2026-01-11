import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Config
from pipeline import run_all


def main() -> None:
    config = Config()
    run_all(config)


if __name__ == "__main__":
    main()
