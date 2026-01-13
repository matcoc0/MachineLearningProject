import sys
from pathlib import Path
import random
import numpy as np
import os
# Add /src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Config
from pipeline import run_all


def set_global_seed(seed: int) -> None:
    

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    config = Config()
    set_global_seed(config.random_seed)

    # Run main (final) pipeline
    run_all(config)




if __name__ == "__main__":
    main()
