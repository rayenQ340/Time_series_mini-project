# =====================================================================================
# Section 3.2 (R) : VAR, causalité de Granger (Toda-Yamamoto), Johansen, VECM — équivalent de 03_var_vecm.ipynb
# Lancer depuis le dossier du projet :  Rscript R/03_var_vecm.R
# Le package `vars` n'est pas nécessaire : le VAR est estimé équation par équation (MCO), ce qui est
# identique à l'estimation d'un VAR (mêmes régresseurs dans chaque équation). Johansen/VECM : package urca.
# =====================================================================================
suppressPackageStartupMessages({ library(urca); library(tseries); library(lmtest); library(lubridate) })
set.seed(42)
df <- read.csv("data/monthly_dataset.csv"); df$month <- as.Date(df$month); df$ly <- log(df$core)
n_train <- floor(0.9 * nrow(df)); train <- df[1:n_train, ]; test <- df[(n_train + 1):nrow(df), ]; H <- nrow(test)
TT <- "t_max_mean"
clim <- function(v) tapply(train[[v]], month(train$month), mean)                 # profil saisonnier : TRAIN uniquement
des  <- function(x, v) x - clim(v)[as.character(month(train$month))]            # moyenne mensuelle retirée

# ---- 1. Stationnarité de la paire (ADF avec retard AIC + KPSS) ---------------------------------
stat <- function(s, name, trend) {
  a <- ur.df(as.numeric(s), type = if (trend) "trend" else "drift", lags = 14, selectlags = "AIC")
  st <- a@teststat[1, if (trend) "tau3" else "tau2"]; cv <- a@cval[if (trend) "tau3" else "tau2", "5pct"]
  k <- suppressWarnings(kpss.test(as.numeric(s), null = if (trend) "Trend" else "Level", lshort = FALSE))
  data.frame(serie = name, spec = if (trend) "const+tend" else "const", ADF = round(st, 2), crit5 = round(cv, 2), KPSS_p = k$p.value)
}
lyd <- as.numeric(des(train$ly, "ly")); Td <- as.numeric(des(train[[TT]], TT))
train$ly_ <- train$ly; clim_ly <- tapply(train$ly, month(train$month), mean); lyd <- train$ly - clim_ly[as.character(month(train$month))]
cat("Stationnarité (séries désaisonnalisées) :\n")
print(rbind(stat(lyd, "log core", FALSE), stat(lyd, "log core", TRUE), stat(Td, TT, FALSE), stat(Td, TT, TRUE),
            stat(diff(lyd), "d log core", FALSE), stat(diff(Td), paste0("d ", TT), FALSE)), row.names = FALSE)
cat(sprintf("pente de la température max désaisonnalisée : %+.3f °C/an\n", coef(lm(Td ~ seq_along(Td)))[2] * 12))

# ---- 2. VAR en niveaux, ordre [température, log production], déterministes : dummies mensuelles + tendance + COVID ----
exog <- function(d, offset = 0) {
  M <- sapply(2:12, function(i) as.numeric(month(d$month) == i)); colnames(M) <- paste0("m", 2:12)
  cbind(M, trend = (seq_len(nrow(d)) - 1 + offset) / 12, covid = d$covid_2020_04)
}
Y <- cbind(temp = train[[TT]], ly = train$ly); EX <- exog(train)
lagmat <- function(Y, p) do.call(cbind, lapply(1:p, function(l) { z <- rbind(matrix(NA, l, ncol(Y)), Y[1:(nrow(Y) - l), , drop = FALSE]); colnames(z) <- paste0(colnames(Y), "_l", l); z }))
eq_data <- function(p, extra, v) {                       # données de l'équation v : variable dépendante + retards + déterministes
  L <- lagmat(Y, p + extra); D <- data.frame(L, EX); D <- D[(p + extra + 1):nrow(D), ]
  D$.y <- Y[(p + extra + 1):nrow(Y), v]; D }
var_fit <- function(p, extra = 0) {                      # extra = retards supplémentaires (Toda-Yamamoto)
  eq <- lapply(colnames(Y), function(v) lm(.y ~ ., data = eq_data(p, extra, v))); names(eq) <- colnames(Y)
  list(eq = eq, U = sapply(eq, resid), p = p, nobs = nrow(eq_data(p, extra, "ly")), k = ncol(EX))
}
info <- function(f) { n <- f$nobs; S <- crossprod(f$U) / n; free <- f$p * 4 + 2 * (f$k + 1)
  c(aic = log(det(S)) + 2 * free / n, bic = log(det(S)) + log(n) * free / n) }
portmanteau <- function(f, h) {                         # Portmanteau multivarié (même formule que statsmodels, non ajusté)
  U <- f$U; n <- nrow(U); C0 <- crossprod(U) / n; C0i <- solve(C0)
  Q <- n * sum(sapply(1:h, function(i) { Ci <- crossprod(U[(i + 1):n, ], U[1:(n - i), ]) / n; sum(diag(t(Ci) %*% C0i %*% Ci %*% C0i)) }))
  c(Q = Q, p = pchisq(Q, 4 * (h - f$p), lower.tail = FALSE)) }
ic <- t(sapply(1:8, function(p) { f <- var_fit(p); c(p = p, info(f), portmanteau_p = unname(portmanteau(f, p + 24)["p"])) }))
cat("\nChoix de l'ordre du VAR :\n"); print(round(ic, 4), row.names = FALSE)
P <- 2; var <- var_fit(P)                                # plus petit ordre à résidus blancs (voir notebook)
cat(sprintf("\nVAR(%d) : Portmanteau p = %.3f | corr. des résidus = %.3f\n", P, portmanteau(var, P + 24)["p"], cor(var$U)[1, 2]))
cat("Jarque-Bera par équation (p) :", round(sapply(1:2, function(i) jarque.bera.test(var$U[, i])$p.value), 3), "\n")
A1 <- rbind(sapply(var$eq, function(m) coef(m)[c("temp_l1", "ly_l1", "temp_l2", "ly_l2")])); cat("Coefficients retardés :\n"); print(round(A1, 4))
comp <- matrix(0, 4, 4); comp[1:2, ] <- cbind(rbind(A1["temp_l1", ], A1["ly_l1", ]) |> t(), rbind(A1["temp_l2", ], A1["ly_l2", ]) |> t()); comp[3:4, 1:2] <- diag(2)
cat("module max des valeurs propres de la matrice compagnon :", round(max(Mod(eigen(comp)$values)), 3), "(< 1 : stable)\n")

# ---- 3. Granger (Toda-Yamamoto) : Wald χ² sur les p premiers retards, VAR(p+1) estimé ------------------
ty <- function(dep, cause, p) {
  d <- eq_data(p, 1, dep); drop <- paste0(cause, "_l", 1:p)
  m  <- lm(.y ~ ., data = d); mr <- lm(.y ~ ., data = d[, !(names(d) %in% drop)])   # restreint : p premiers retards de `cause` retirés
  w <- waldtest(m, mr, test = "Chisq"); c(chi2 = w$Chisq[2], p = w$`Pr(>Chisq)`[2]) }
gr <- do.call(rbind, lapply(1:6, function(p) { a <- ty("ly", "temp", p); b <- ty("temp", "ly", p)
  data.frame(p = p, T_to_prod_chi2 = a["chi2"], T_to_prod_p = a["p"], prod_to_T_chi2 = b["chi2"], prod_to_T_p = b["p"], row.names = NULL) }))
cat("\nCausalité de Granger (Toda-Yamamoto) :\n"); print(round(gr, 3), row.names = FALSE)
ct <- cor.test(var$U[, 1], var$U[, 2]); cat(sprintf("Causalité instantanée (test de corrélation des résidus) : r = %.3f, p = %.2g\n", ct$estimate, ct$p.value))

# ---- 4. Cointégration (séries désaisonnalisées) -------------------------------------------------
D <- cbind(ly = lyd, temp = Td)
rk <- function(j) { r <- 0; while (r < 2 && j@teststat[2 - r] > j@cval[2 - r, "5pct"]) r <- r + 1; r }   # teststat est classé r<=1 puis r=0
rows <- NULL
for (ec in c("none", "const")) for (k in c(1, 2, 3, 6)) {
  j <- ca.jo(D, type = "trace", ecdet = ec, K = k + 1, spec = "longrun")
  rows <- rbind(rows, data.frame(ecdet = ec, k = k, trace_r0 = j@teststat[2], cv_r0 = j@cval[2, "5pct"],
                                 trace_r1 = j@teststat[1], cv_r1 = j@cval[1, "5pct"], rang = rk(j)))
}
cat("\nJohansen (trace) :\n"); print(round(rows[, -1], 1) |> cbind(ecdet = rows$ecdet), row.names = FALSE)
# Engle-Granger : MCO de y sur x (const. ou const.+tend.), puis ADF (sans constante) sur les résidus ; on compare à la valeur
# critique asymptotique de MacKinnon à 5 % pour 2 variables (-3,34 const. ; -3,78 const.+tend.) — les p-values usuelles ne sont pas celles d'un ADF.
eg <- function(y, x, trend) {
  r <- resid(if (trend) lm(y ~ x + seq_along(y)) else lm(y ~ x))
  a <- ur.df(r, type = "none", lags = 14, selectlags = "AIC"); st <- a@teststat[1, "tau1"]; cv <- if (trend) -3.78 else -3.34
  c(stat = round(st, 2), crit5 = cv, rejette_H0_non_coint = st < cv) }
cat("\nEngle-Granger (statistique ADF des résidus vs valeur critique 5 %) :\n")
print(rbind("ly ~ T, const." = eg(D[, "ly"], D[, "temp"], FALSE), "ly ~ T, const.+tend." = eg(D[, "ly"], D[, "temp"], TRUE),
            "T ~ ly, const." = eg(D[, "temp"], D[, "ly"], FALSE), "T ~ ly, const.+tend." = eg(D[, "temp"], D[, "ly"], TRUE)))
cat("Phillips-Ouliaris (test alternatif, p-values) — ly ~ T :", round(suppressWarnings(po.test(cbind(D[, "ly"], D[, "temp"]))$p.value), 3),
    "| T ~ ly :", round(suppressWarnings(po.test(cbind(D[, "temp"], D[, "ly"]))$p.value), 3), "\n")

# ---- 5. VECM (rang 1) -----------------------------------------------------------------------------
j1 <- ca.jo(D, type = "trace", ecdet = "const", K = 2, spec = "longrun"); vm <- cajorls(j1, r = 1)
beta <- vm$beta[, 1]; cat(sprintf("\nVECM : vecteur normalisé sur ly : beta = (1 ; %+.3f)\n", beta[2] / beta[1]))
sm <- lapply(summary(vm$rlm), function(s) s$coefficients["ect1", c("Estimate", "Pr(>|t|)")])
cat("alpha (vitesse d'ajustement), p-value :\n"); print(round(do.call(rbind, sm), 4))

# ---- 6. Prévisions h = 1..24 du VAR en niveaux (aucune météo future utilisée) --------------------------
EXf <- exog(test, offset = nrow(train)); hist <- Y; fc <- matrix(NA, H, 2)
for (h in 1:H) {
  x <- c(1, hist[nrow(hist), ], hist[nrow(hist) - 1, ], EXf[h, ])                         # ordre : const, lag1 (temp, ly), lag2 (temp, ly), exog
  nm <- c("(Intercept)", "temp_l1", "ly_l1", "temp_l2", "ly_l2", colnames(EX))
  fc[h, ] <- sapply(var$eq, function(m) sum(coef(m)[nm] * x)); hist <- rbind(hist, fc[h, ])
}
pred <- exp(fc[, 2]); metr <- function(a, p, h) { e <- (a - p)[1:h]; c(RMSE = sqrt(mean(e^2)), MAE = mean(abs(e)), MAPE = 100 * mean(abs(e) / a[1:h])) }
res <- rbind(h10 = metr(test$core, pred, 10), h24 = metr(test$core, pred, 24)); cat("\nVAR(2) niveaux — métriques test :\n"); print(round(res, 1))
dir.create("results", showWarnings = FALSE)
write.csv(data.frame(month = format(test$month), VAR2_niveaux = round(pred, 3)), "results/forecasts_var_R.csv", row.names = FALSE)
