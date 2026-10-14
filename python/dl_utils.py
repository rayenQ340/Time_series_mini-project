"""LSTM / GRU pour la section 4.1 : fenêtres glissantes, entraînement, prévision récursive.

Cible : z_t = log(core_t) - log(core_{t-12})  (variation annuelle en log). Un réseau de neurones ne sait pas extrapoler une
tendance ; en prédisant la variation annuelle on garde le niveau et la saisonnalité de l'année précédente (comme le naïf saisonnier)
et le réseau n'apprend que l'écart de croissance. Le niveau est reconstruit : log(core_t) = log(core_{t-12}) + z_t.
Entrées à chaque pas s (NARX) : z_{s-1} (cible retardée), sin/cos du mois, indicatrice COVID annualisée, et (optionnel) la variation
annuelle des degrés-jours de climatisation dcdd_s = cdd24_s - cdd24_{s-12} (connue au pas s : météo réelle ou climatologique).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3"); os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np, pandas as pd, tensorflow as tf

FEATS = {"univarié": ["zlag", "sin", "cos", "cov12"],
         "avec chaleur": ["zlag", "dcdd", "sin", "cos", "cov12"]}

def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def static_features(index, cdd, covid):
    """Entrées qui ne dépendent pas de la cible : saisonnalité, COVID annualisé, variation annuelle de la chaleur."""
    s = pd.DataFrame(index=index)
    s["sin"] = np.sin(2 * np.pi * index.month / 12); s["cos"] = np.cos(2 * np.pi * index.month / 12)
    c = pd.Series(covid, index=index, dtype=float); s["cov12"] = (c - c.shift(12)).fillna(0.0)
    cd = pd.Series(cdd, index=index, dtype=float); s["dcdd"] = cd - cd.shift(12)
    return s

def cdd_climatology(df_known, index_all):
    """cdd24 réel jusqu'à l'origine, puis moyenne mensuelle de l'historique connu (version réalisable : aucune météo future)."""
    clim = df_known.groupby(df_known.index.month)["cdd24"].mean()
    out = pd.Series(index=index_all, dtype=float)
    out.loc[df_known.index] = df_known["cdd24"].values
    for t in index_all.difference(df_known.index): out.loc[t] = clim.loc[t.month]
    return out

def training_frame(ly, static):
    D = static.copy(); z = ly.diff(12); D["z"] = z; D["zlag"] = z.shift(1)
    return D.loc[ly.index].dropna(subset=["z", "zlag", "dcdd"])

def windows(X, y, w):
    Xs = np.stack([X[i - w + 1:i + 1] for i in range(w - 1, len(X))]); return Xs, y[w - 1:]

def build_model(arch, w, nf, units, dropout, lr=1e-2):
    layer = tf.keras.layers.LSTM if arch == "LSTM" else tf.keras.layers.GRU
    m = tf.keras.Sequential([tf.keras.Input((w, nf)), layer(units), tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(1)])
    m.compile(tf.keras.optimizers.Adam(lr), "mse"); return m

def fit_nn(arch, featset, ly, static, w, units, dropout, seed, epochs=None, n_val=24, patience=15, lr=1e-2, min_epochs=5):
    """epochs=None : mode sélection (les n_val dernières fenêtres, CHRONOLOGIQUEMENT, servent de validation + arrêt précoce).
       epochs=k     : mode final (toutes les fenêtres, k époques fixées)."""
    feats = FEATS[featset]; D = training_frame(ly, static)
    mu, sd = D[feats + ["z"]].mean(), D[feats + ["z"]].std().replace(0, 1.0)         # normalisation estimée sur l'apprentissage seul
    S = (D[feats + ["z"]] - mu) / sd
    X, y = windows(S[feats].values, S["z"].values, w)
    set_seed(seed); m = build_model(arch, w, len(feats), units, dropout, lr)
    if epochs is None:
        h = m.fit(X[:-n_val], y[:-n_val], validation_data=(X[-n_val:], y[-n_val:]), epochs=200, batch_size=16, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, start_from_epoch=min_epochs)])
        hist = h.history; best = int(np.argmin(hist["val_loss"][min_epochs:])) + min_epochs + 1   # au moins min_epochs époques d'entraînement
    else:
        h = m.fit(X, y, epochs=epochs, batch_size=16, verbose=0); hist, best = h.history, epochs
    return dict(model=m, feats=feats, mu=mu, sd=sd, w=w, history=hist, best_epoch=best, arch=arch, featset=featset, val_loss=(min(hist["val_loss"]) if "val_loss" in hist else np.nan),
                val_baseline=(float(np.mean(y[-n_val:] ** 2)) if epochs is None else np.nan))   # MSE de la prédiction constante (moyenne du train) sur la validation

def forecast_recursive(predict_z, ly_known, static_all, feats, w, H):
    """Prévision récursive à H pas. predict_z(window[w, nf]) -> z_t (échelle d'origine). static_all : index = connu + futur."""
    n0 = len(ly_known); idx = static_all.index[:n0 + H]
    ly = np.full(n0 + H, np.nan); ly[:n0] = ly_known.values
    z = np.full(n0 + H, np.nan); z[12:n0] = ly[12:n0] - ly[:n0 - 12]
    st = static_all.loc[idx]
    for k in range(H):
        t = n0 + k; rows = []
        for s in range(t - w + 1, t + 1):
            r = {"zlag": z[s - 1], "dcdd": st["dcdd"].iloc[s], "sin": st["sin"].iloc[s], "cos": st["cos"].iloc[s], "cov12": st["cov12"].iloc[s]}
            rows.append([r[f] for f in feats])
        z[t] = predict_z(np.asarray(rows, float)); ly[t] = ly[t - 12] + z[t]
    return pd.Series(np.exp(ly[n0:]), index=idx[n0:])

def nn_predictor(info):
    def f(win):
        x = ((pd.DataFrame(win, columns=info["feats"]) - info["mu"][info["feats"]]) / info["sd"][info["feats"]]).values[None]
        return float(info["model"](x, training=False).numpy()[0, 0] * info["sd"]["z"] + info["mu"]["z"])
    return f

def fit_ols(ly, static, featset):
    """Référence linéaire avec les mêmes entrées (fenêtre de 1 pas) : z_t = b0 + b'x_t."""
    feats = FEATS[featset]; D = training_frame(ly, static)
    A = np.column_stack([np.ones(len(D)), D[feats].values]); beta = np.linalg.lstsq(A, D["z"].values, rcond=None)[0]
    return dict(beta=beta, feats=feats, w=1)

def ols_predictor(info):
    return lambda win: float(info["beta"][0] + win[-1] @ info["beta"][1:])
