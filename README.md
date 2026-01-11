# MachineLearningProject

Console-based project for the Higgs Boson detection assignment using:
- Genetic Programming (GA) baseline
- GA + Active Learning (GA+AL)
- GA + Active Learning + Ensemble Learning (GA+AL+EL)

All code lives under `src/` and running the project generates reports under `reports/`.

## Requirements
- Python 3.12
- Dataset file: `data/atlas-higgs-challenge-2014-v2.csv.gz`

Install dependencies:

```bash
python3.12 -m pip install -r requirements.txt
```

## Run

```bash
python3.12 main.py
```

The script will:
1. Load and preprocess the dataset (imputation + standardization).
2. Train GA, GA+AL, and GA+AL+EL models.
3. Print metrics to the console.
4. Save reports to `reports/`.

## Outputs
The following files are generated (overwritten on each run):
- `reports/metrics.csv` / `reports/metrics.json`
- `reports/metrics_plot.png`
- `reports/training_time.png`
- `reports/ga_log.csv`
- `reports/ga_al_log.csv`
- `reports/ga_al_el_log.csv`

## Configuration
You can edit settings in `src/config.py`, including:
- Number of generations
- Population size and genetic operators
- Active learning sampling settings
- Ensemble size
