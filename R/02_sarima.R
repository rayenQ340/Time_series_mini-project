# =====================================================================================
# Section 3.1 (R) : ARIMA / SARIMA / SARIMAX — équivalent de 02_sarima.ipynb
# Lancer depuis le dossier du projet :  Rscript R/02_sarima.R
# Prérequis : data/monthly_dataset.csv (produit par 01_eda.ipynb ou R/01_eda.R -> data/monthly_dataset_R.csv, identique)
# =====================================================================================
suppressPackageStartupMessages({ library(forecast); library(tseries); library(lubridate) })
set.seed(42)                                           # graine fixée
cat("R", R.version.string, "| forecast", as.character(packageVersion("forecast")), "\n")

# ---- Données et découpage chronologique 90/10 ---------------------------------------------
df <- read.csv("data/monthly_dataset.csv"); df$month <- as.Date(df$month)
n_train <- floor(0.9 * nrow(df)); train <- df[1:n_train, ]; test <- df[(n_train + 1):nrow(df), ]
H <- nrow(test)
ly <- ts(log(train$core), start = c(2005, 1), frequency = 12)    # log : saisonnalité multiplicative (EDA)
X0 <- cbind(covid = train$covid_2020_04)                          # exogène minimale : indicatrice COVID
Xt0 <- cbind(covid = test$covid_2020_04)

# ---- Ajustement : méthode ML, sans contraintes de stationnarité imposées par l'optimiseur ----
fit <- function(order, seas, xreg = NULL)
  Arima(ly, order = order, seasonal = list(order = seas, period = 12), xreg = xreg, method = "ML")

# ---- 1. Modèle manuel (ACF/PACF de l'EDA : pics négatifs aux retards 1 et 12) --------------
m_manual <- fit(c(0, 1, 1), c(0, 1, 1), X0)
print(m_manual)

# ---- 2. Recherche automatique + grille explicite (AICc, comparable seulement au sein de R) --
m_auto <- auto.arima(ly, xreg = X0, d = 1, D = 1, max.p = 3, max.q = 3, max.P = 2, max.Q = 2,
                     stepwise = FALSE, approximation = FALSE, ic = "aicc")   # recherche exhaustive : plus fiable que stepwise
cat("auto.arima :", arimaorder(m_auto)[c(1, 2, 3)], "| saisonnier", arimaorder(m_auto)[c(4, 5, 6)], "\n")

grid <- function(xreg) {
  out <- NULL
  for (p in 0:2) for (q in 0:2) for (P in 0:1) for (Q in 0:1) {
    m <- tryCatch(fit(c(p, 1, q), c(P, 1, Q), xreg), error = function(e) NULL)
    if (!is.null(m)) out <- rbind(out, data.frame(p, q, P, Q, AICc = m$aicc, BIC = m$bic))
  }
  out <- out[order(out$AICc), ]
  # Règle de parcimonie (identique en Python) : parmi les modèles à moins de 2 points d'AICc du meilleur,
  # on prend celui qui a le moins de paramètres ARMA (puis le plus petit AICc) -> évite de sur-interpréter un écart non significatif.
  near <- out[out$AICc - out$AICc[1] < 2, ]; npar <- near$p + near$q + near$P + near$Q
  out$retenu <- FALSE; out$retenu[rownames(out) == rownames(near[order(npar, near$AICc), ][1, ])] <- TRUE
  out
}
g0 <- grid(X0); cat("\nGrille (exogène = COVID) — 6 meilleurs par AICc :\n"); print(head(g0[, 1:6], 6), row.names = FALSE, digits = 5)

# ---- 3. Diagnostic complet des résidus ------------------------------------------------------
# On écarte les premiers résidus (initialisation du filtre) : d + D*s + p + P*s observations.
arch_lm <- function(r, lags = 12) {                      # ARCH-LM d'Engle : n*R2 de la régression de r^2 sur ses retards
  r2 <- r^2; n <- length(r2); M <- embed(r2, lags + 1)
  reg <- summary(lm(M[, 1] ~ M[, -1])); stat <- (n - lags) * reg$r.squared
  c(stat = stat, p = pchisq(stat, lags, lower.tail = FALSE))
}
diagnose <- function(m, name, order, seas) {
  burn <- order[2] + seas[2] * 12 + order[1] + seas[1] * 12
  r <- as.numeric(residuals(m))[-(1:burn)]; npar <- order[1] + order[3] + seas[1] + seas[3]
  data.frame(modele = name,
             LB12_p = Box.test(r, 12, "Ljung-Box", fitdf = npar)$p.value, LB24_p = Box.test(r, 24, "Ljung-Box", fitdf = npar)$p.value,
             JB_p = jarque.bera.test(r)$p.value, Shapiro_p = shapiro.test(r)$p.value,
             ARCH12_p = arch_lm(r)["p"], sd_resid = sd(r))
}
ao <- arimaorder(m_auto)
diag <- rbind(diagnose(m_manual, "SARIMA (0,1,1)(0,1,1) manuel", c(0, 1, 1), c(0, 1, 1)),
              diagnose(m_auto, sprintf("SARIMA auto (%d,1,%d)(%d,1,%d)", ao[1], ao[3], ao[4], ao[6]), c(ao[1], 1, ao[3]), c(ao[4], 1, ao[6])))

# ---- 4. SARIMAX : la température comme exogène ----------------------------------------------
cands <- list("COVID seul" = "covid_2020_04", "+ days_above_35" = c("covid_2020_04", "days_above_35"),
              "+ cdd24" = c("covid_2020_04", "cdd24"), "+ les deux" = c("covid_2020_04", "days_above_35", "cdd24"))
cmp <- do.call(rbind, lapply(names(cands), function(k) {
  m <- fit(c(0, 1, 1), c(0, 1, 1), as.matrix(train[, cands[[k]], drop = FALSE]))
  data.frame(exogenes = k, AICc = m$aicc, BIC = m$bic)
}))
cat("\nSARIMAX (0,1,1)(0,1,1) selon l'exogène :\n"); print(cmp, row.names = FALSE, digits = 6)
cat("corr(days_above_35, cdd24) =", round(cor(train$days_above_35, train$cdd24), 2), "\n")

EX <- c("covid_2020_04", "cdd24")                           # choix retenu (cdd24 seul : l'autre devient non significatif)
gx <- grid(as.matrix(train[, EX])); cat("\nGrille SARIMAX (COVID + cdd24) — 5 meilleurs :\n"); print(head(gx[, 1:6], 5), row.names = FALSE, digits = 5); cat("modèle retenu par la règle de parcimonie : p,q,P,Q =", unlist(gx[gx$retenu, 1:4]), "\n")
bx <- gx[gx$retenu, ]; ox <- c(bx$p, 1, bx$q); sx <- c(bx$P, 1, bx$Q)
m_x <- fit(ox, sx, as.matrix(train[, EX])); print(m_x)
diag <- rbind(diag, diagnose(m_x, sprintf("SARIMAX (%d,1,%d)(%d,1,%d) + cdd24", ox[1], ox[3], sx[1], sx[3]), ox, sx))
cat("\nDiagnostic des résidus :\n"); print(diag, row.names = FALSE, digits = 3)

# ---- 5. Prévisions h = 1..24 (origine fixe fin août 2022), IC 95 % calculés en log puis re-transformés ----
fc <- function(m, xf) { f <- forecast(m, h = H, xreg = xf, level = 95)
  data.frame(fc = exp(as.numeric(f$mean)), lo = exp(as.numeric(f$lower)), hi = exp(as.numeric(f$upper))) }
clim <- tapply(train$cdd24, month(train$month), mean)       # climatologie : version réalisable sans prévision météo
Xclim <- cbind(covid_2020_04 = 0, cdd24 = as.numeric(clim[as.character(month(test$month))]))
F <- list("Naïf saisonnier" = data.frame(fc = rep(tail(train$core, 12), 2)),
          "SARIMA manuel" = fc(m_manual, Xt0), "SARIMA auto" = fc(m_auto, Xt0),
          "SARIMAX oracle" = fc(m_x, as.matrix(test[, EX])), "SARIMAX climatologie" = fc(m_x, Xclim))
metr <- function(a, p, h) { e <- (a - p)[1:h]; c(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), MAPE = 100 * mean(abs(e) / a[1:h])) }
res <- do.call(rbind, lapply(names(F), function(n) { f <- F[[n]]
  a <- metr(test$core, f$fc, 10); b <- metr(test$core, f$fc, 24)
  data.frame(modele = n, RMSE_h10 = a["RMSE"], MAE_h10 = a["MAE"], MAPE_h10 = a["MAPE"],
             RMSE_h24 = b["RMSE"], MAE_h24 = b["MAE"], MAPE_h24 = b["MAPE"],
             couverture_IC95 = if ("lo" %in% names(f)) 100 * mean(test$core >= f$lo & test$core <= f$hi) else NA, row.names = NULL) }))
cat("\nMétriques sur l'échantillon test :\n"); print(res, row.names = FALSE, digits = 4)

dir.create("results", showWarnings = FALSE)
write.csv(cbind(month = format(test$month), as.data.frame(lapply(F, function(f) round(f$fc, 3)), check.names = FALSE)),
          "results/forecasts_R.csv", row.names = FALSE)
write.csv(res, "results/metrics_sarima_R.csv", row.names = FALSE)
