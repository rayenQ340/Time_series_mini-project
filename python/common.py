"""Fonctions communes à tous les notebooks (une seule définition => mêmes métriques partout)."""
import os, numpy as np, pandas as pd

DATA_PATH = "data/monthly_dataset.csv"      # produit par 01_eda.ipynb
RESULTS_DIR = "results"

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
    """Ajoute/écrase la colonne `name` dans results/forecasts.csv (utilisé plus tard pour Diebold-Mariano)."""
    os.makedirs(RESULTS_DIR, exist_ok=True); path = os.path.join(RESULTS_DIR, fname)
    s = pd.Series(np.asarray(series, float), index=series.index, name=name)
    out = pd.read_csv(path, index_col=0, parse_dates=True) if os.path.exists(path) else pd.DataFrame(index=s.index)
    out[name] = s; out.to_csv(path)
