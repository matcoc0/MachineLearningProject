from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/atlas-higgs-challenge-2014-v2.csv.gz")
    reports_dir: Path = Path("reports")
    random_seed: int = 42
    test_size: float = 0.2

    # Genetic programming settings
    population_size: int = 200
    generations: int = 30
    crossover_prob: float = 0.5
    mutation_prob: float = 0.3
    tournament_size: int = 5
    max_tree_height: int = 6

    # Active learning settings
    al_enabled: bool = True
    al_initial_fraction: float = 0.1
    al_samples_per_round: int = 5000
    al_interval: int = 2
    al_strategy: str = "uncertainty"  # uncertainty or random

    # Ensemble settings
    ensemble_size: int = 15
