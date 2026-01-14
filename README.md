# MachineLearningProject
Authored by Mathieu Cowan and Hugo Bonnell

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
or
```bash
python -m pip install -r requirements.txt
```


## Run

```bash
python3.12 main.py
```
or
```bash
python main.py
```

The script will:
1. Load and preprocess the dataset (imputation + standardization).
2. Train GA, GA+AL, and GA+AL+EL models.
3. Print metrics to the console.
4. Save reports to `reports/`.

## Folder Description
- **`data/`**  
  Contains the raw Higgs Boson dataset used for training and evaluation.

- **`src/`**  
  Main source code of the project:
  - `config.py`: global configuration and hyperparameters
  - `data.py`: data loading and preprocessing
  - `eda.py`: exploratory data analysis
  - `ga.py`: Genetic Programming implementation (GA and GA+AL)
  - `ensemble.py`: ensemble prediction strategies
  - `experiments.py`: experimental variants and ablation studies
  - `metrics.py`: evaluation metrics and prediction helpers
  - `reporting.py`: logging, plotting and report generation
  - `pipeline.py`: main orchestration logic

- **`reports/`**  
  Automatically generated outputs:
  - `reports/data/`: EDA outputs (statistics, correlations, PCA)
  - `reports/training/`: GA and GA+AL training logs and convergence plots
  - `reports/results/`: main performance metrics and training time comparison
  - `reports/results/experiments/`: experimental variant results and plots

---

### Notes

- All results are **overwritten on each run** to ensure reproducibility.
- Experimental variants are executed only if `run_experiments=True` in `config.py`.
- Logs and plots explicitly indicate the trained approach (GA, GA+AL, GA+AL+EL).

## Outputs

### reports/
- `metrics.csv` – Aggregated classification metrics for all approaches  
- `metrics.json` – Metrics stored in structured JSON format  
- `metrics_plot.png` – Comparative plot of accuracy, precision, recall and F1-score  
- `training_time.png` – Training time comparison across models  

### reports/training/
- `ga_log.csv` – GA training evolution (best and mean F1 per generation)  
- `ga_al_log.csv` – GA + Active Learning training evolution  
- `ga_al_el_log.csv` – Ensemble-level performance evolution  

### reports/results/
- `metrics.csv` – Final evaluation metrics on the test set  
- `metrics.json` – Serialized results for reproducibility  

### reports/results/experiments/
- `*.csv` – Metrics for experimental model variants  
- `*.png` – Performance comparison plots for experiments 

## Configuration
You can edit settings in `src/config.py`, including:
- Number of generations
- Population size and genetic operators
- Active learning sampling settings
- Ensemble size
- Activation true/false of the run of the EDA : run_eda
- Activation true/false of the run of the experimental models after baseline model : run_experiments
  Actual state : run_eda = true, run_experiments = false
