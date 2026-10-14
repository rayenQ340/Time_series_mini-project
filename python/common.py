"""Fonctions communes à tous les notebooks (une seule définition => mêmes métriques partout).

Chemins : ils sont calculés à partir de l'emplacement de CE fichier (python/ -> racine du projet), puis le dossier de travail
est ramené à la racine dès l'import. Les notebooks (lancés depuis python/, là où se trouve common.py) et les chemins relatifs
« data/... » et « results/... » qu'ils utilisent fonctionnent donc quel que soit le dossier d'où Jupyter est démarré.
"""
import os, random
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent            # racine du projet (contient data/, results/, python/, R/)
os.chdir(ROOT)                                           # rend valides les chemins relatifs "data/..." et "results/..." des notebooks
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "monthly_dataset.csv"             # produit par 01_eda.ipynb
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42

def set_seed(seed=SEED):
    """Graine fixée pour random et numpy (reproductibilité). TensorFlow est initialisé séparément dans dl_utils.set_seed."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed)); random.seed(seed); np.random.seed(seed)

def load():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True); df.index.freq = "MS"
    return df

def split(df, frac=0.9):
    """Découpage chronologique 90/10 (jamais aléatoire)."""
    n = int(frac * len(df)); return df.iloc[:n].copy(), df.iloc[n:].copy()

def metrics(actual, pred, h=None):
    """MSE, RMSE, MAE, MAPE (en %) sur les h premiers pas de l'échantillon test (échelle GWh)."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    if h: a, p = a[:h], p[:h]
    e = a - p
    return {"MSE": np.mean(e**2), "RMSE": np.sqrt(np.mean(e**2)), "MAE": np.mean(np.abs(e)),
            "MAPE": 100 * np.mean(np.abs(e) / a)}

def save_forecasts(name, series, fname="forecasts.csv"):
    """Ajoute/écrase la colonne `name` dans results/forecasts.csv (utilisé pour le tableau final et Diebold-Mariano)."""
    path = RESULTS_DIR / fname
    s = pd.Series(np.asarray(series, float), index=series.index, name=name)
    out = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else pd.DataFrame(index=s.index)
    out[name] = s; out.to_csv(path)

def load_forecasts(fname="forecasts.csv"):
    """Relit toutes les prévisions sauvegardées (une colonne par modèle, échelle GWh)."""
    return pd.read_csv(RESULTS_DIR / fname, index_col=0, parse_dates=True)

def diebold_mariano(actual, f1, f2, h=1, loss="se"):
    """Test de Diebold-Mariano avec correction de Harvey-Leybourne-Newbold (loi t(n-1)).

    H0 : même précision de prévision. Différentiel de perte d_t = L(e1_t) - L(e2_t) ; statistique < 0 => f1 meilleur.
    `h` = horizon des prévisions (la variance de long terme utilise h-1 autocovariances) ; `loss` = "se" (carré) ou "ae" (absolu).
    Retourne (statistique, p-value bilatérale). Avec peu d'observations (ex. 24) la puissance du test est faible.
    """
    from scipy import stats
    a, p1, p2 = (np.asarray(x, float) for x in (actual, f1, f2))
    e1, e2 = a - p1, a - p2
    d = e1**2 - e2**2 if loss == "se" else np.abs(e1) - np.abs(e2)
    n = len(d); dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0: return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)          # correction petit échantillon (HLN 1997)
    return hln, 2 * stats.t.sf(abs(hln), df=n - 1)