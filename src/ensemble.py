from __future__ import annotations

import numpy as np
from deap import tools


def ensemble_predict(population, toolbox, x, ensemble_size):
    top = tools.selBest(population, min(ensemble_size, len(population)))
    scores = []
    for ind in top:
        func = toolbox.compile(expr=ind)
        outputs = func(*x.T)
        outputs = np.asarray(outputs)
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
        scores.append(outputs)

    avg_scores = np.mean(np.vstack(scores), axis=0)
    preds = (avg_scores > 0).astype(int)
    return preds
