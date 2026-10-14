# =====================================================================================
# Section 3.3 (R) : ARCH / GARCH — équivalent de 04_garch.ipynb
# Lancer depuis le dossier du projet :  Rscript R/04_garch.R
# Package GARCH : fGarch (cité dans l'énoncé ; rugarch n'est pas requis). Prérequis : forecast, tseries, fGarch.
# =====================================================================================
suppressPackageStartupMessages({ library(forecast); library(tseries); library(fGarch); library(lubridate) })
set.seed(42)
df <- read.csv("data/monthly_dataset.csv"); df$month <- as.Date(df$month)
n_train <- floor(0.9 * nrow(df)); train <- df[1:n_train, ]
ly <- ts(log(train$core), start = c(2005, 1), frequency = 12)
cat("R", R.version.string, "| fGarch", as.character(packageVersion("fGarch")), "\n")

# ---- Résidus des deux modèles de la section 3.1 (en %), initialisation du filtre écartée ----
fit <- function(order, seas, xreg) Arima(ly, order = order, seasonal = list(order = seas, period = 12), xreg = xreg, method = "ML")
m_sar <- fit(c(0, 1, 1), c(0, 1, 1), cbind(covid = train$covid_2020_04))
m_x   <- fit(c(1, 1, 1), c(0, 1, 1), cbind(covid = train$covid_2020_04, cdd24 = train$cdd24))
r  <- 100 * as.numeric(residuals(m_sar))[-(1:13)]       # burn-in = d + D*s = 13
rx <- 100 * as.numeric(residuals(m_x))[-(1:14)]         # + p = 1
mo <- month(train$month)[-(1:13)]; mox <- month(train$month)[-(1:14)]

# ---- 1. Tests ARCH (ARCH-LM d'Engle : n*R2 de r^2 sur ses retards) + variance saisonnière ----
arch_lm <- function(x, lags) { x2 <- x^2; M <- embed(x2, lags + 1); R2 <- summary(lm(M[, 1] ~ M[, -1]))$r.squared
  pchisq((length(x2) - lags) * R2, lags, lower.tail = FALSE) }
tests <- rbind(SARIMA = sapply(c(1, 3, 6, 12), function(L) arch_lm(r, L)), SARIMAX = sapply(c(1, 3, 6, 12), function(L) arch_lm(rx, L)))
colnames(tests) <- paste0("ARCH-LM(", c(1, 3, 6, 12), ") p"); cat("\nTests ARCH :\n"); print(round(tests, 4))
sdm <- function(x, m) sqrt(tapply(x^2, m, mean))
sd_s <- sdm(r, mo); sd_x <- sdm(rx, mox)
cat("écart-type par mois (SARIMA) :", round(sd_s, 2), "\n")
cat(sprintf("ratio écart-type juil.-sept. / avr.-mai — SARIMA : %.2f | SARIMAX : %.2f\n",
            mean(sd_s[c("7", "8", "9")]) / mean(sd_s[c("4", "5")]), mean(sd_x[c("7", "8", "9")]) / mean(sd_x[c("4", "5")])))

# ---- 2. Famille GARCH (fGarch::garchFit, moyenne nulle) ----
gfit <- function(p, q, cond = "norm", y = r)
  suppressWarnings(garchFit(as.formula(sprintf("~ garch(%d, %d)", p, q)), data = y, cond.dist = cond, include.mean = FALSE, trace = FALSE))
diagz <- function(f, name) { z <- as.numeric(residuals(f, standardize = TRUE)); ic <- f@fit$ics
  data.frame(modele = name, AIC = round(ic["AIC"] * length(z), 1), BIC = round(ic["BIC"] * length(z), 1),   # fGarch donne AIC/BIC par observation
             LBz_p = Box.test(z, 12, "Ljung-Box")$p.value, LBz2_p = Box.test(z^2, 12, "Ljung-Box")$p.value,
             ARCH12_p = arch_lm(z, 12), JB_p = jarque.bera.test(z)$p.value,
             alpha = sum(coef(f)[grep("^alpha", names(coef(f)))]), beta = sum(coef(f)[grep("^beta", names(coef(f)))]), row.names = NULL) }
# ATTENTION : garchFit ajuste le GARCH(1,1) sur un optimum local (alpha~0,06 ; beta~0,91) dont l'AIC est PIRE que celui de l'ARCH(1) qu'il contient
# comme cas particulier. On ajuste donc l'ARCH(1) explicitement (garch(1,0)) et on le retient (voir 04_garch.ipynb).
specs <- list("ARCH(1) norm" = c(1, 0, "norm"), "GARCH(1,1) norm" = c(1, 1, "norm"), "GARCH(1,1) Student" = c(1, 1, "std"), "GARCH(1,2) norm" = c(1, 2, "norm"),
              "GARCH(2,1) norm" = c(2, 1, "norm"), "GARCH(2,2) norm" = c(2, 2, "norm"))
fits <- lapply(specs, function(s) gfit(as.integer(s[1]), as.integer(s[2]), s[3]))
tab <- do.call(rbind, Map(diagz, fits, names(fits))); cat("\nFamille GARCH :\n"); print(tab, row.names = FALSE, digits = 3)

# ---- 3. Modèle retenu : diagnostic + volatilité conditionnelle ----
best <- "ARCH(1) norm"; f <- fits[[best]]; z <- as.numeric(residuals(f, standardize = TRUE))
cat("\nParamètres du modèle retenu (", best, ") :\n", sep = ""); print(round(f@fit$matcoef, 4))
cat(sprintf("asymétrie de z = %.2f | Jarque-Bera p = %.4f | Shapiro p = %.4f\n", (mean((z - mean(z))^3) / sd(z)^3), jarque.bera.test(z)$p.value, shapiro.test(z)$p.value))
cv <- f@sigma.t
png("figures_R/04_garch_vol.png", 1100, 450, res = 110); plot(r_idx <- seq_along(cv), abs(r), type = "l", col = "gray70", xlab = "mois (depuis 2006-02)", ylab = "%")
lines(cv, col = "red", lwd = 2); legend("topleft", c("|résidu|", "volatilité conditionnelle"), col = c("gray70", "red"), lty = 1, bty = "n"); dev.off()
dates <- train$month[-(1:13)]; top <- order(-cv)[1:8]
cat("\nMois de plus forte volatilité conditionnelle :\n")
print(data.frame(mois = format(dates[top], "%Y-%m"), vol_pct = round(cv[top], 1), choc_mois_precedent = round(r[top - 1], 1)), row.names = FALSE)

# ---- 4. Robustesse : variance saisonnière ----
zs <- as.numeric(r / sqrt(tapply(r^2, mo, mean)[as.character(mo)]))
cat("\nARCH-LM sur résidus désaisonnalisés en variance :", round(sapply(c(1, 3, 6, 12), function(L) arch_lm(zs, L)), 3), "\n")
cat("ARCH-LM sur résidus bruts                        :", round(sapply(c(1, 3, 6, 12), function(L) arch_lm(r, L)), 3), "\n")
