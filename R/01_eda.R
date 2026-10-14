# =====================================================================================
# Mini-projet Séries Temporelles — Partie 1 (R) : données & analyse exploratoire
# Équivalent R de 01_eda.ipynb. À lancer depuis le dossier du projet (là où se trouve data/) :
#     Rscript R/01_eda.R
# Packages : readxl, lubridate, dplyr, forecast, tseries, urca, zoo   (voir R/00_packages.R)
# Chaque bloc dit POURQUOI on fait ce choix (exigence de l'énoncé), pas seulement ce que fait le code.
# =====================================================================================
suppressPackageStartupMessages({
  library(readxl); library(lubridate); library(dplyr)
  library(forecast); library(tseries); library(urca)
})
set.seed(42)                                   # graine fixée -> reproductibilité
dir.create("figures_R", showWarnings = FALSE)
cat("R", R.version.string, "| forecast", as.character(packageVersion("forecast")),
    "| tseries", as.character(packageVersion("tseries")), "| urca", as.character(packageVersion("urca")), "\n")

# ---------------------------------------------------------------- 1. DONNÉES --------------
# Production : malgré son extension, le fichier est un ancien format .xls -> read_xls() (read_excel() se fie à l'extension et échoue).
prod <- read_xls("data/opendataproductionmensuelleelectriciteparproducteurs (1).xlsx",
                   col_types = "text")                   # tout en texte : la colonne date est mixte
names(prod) <- c("date", "steg", "ipp", "ipp_solar", "auto", "total_raw")

# Les dates sont mixtes (texte "m / aaaa" puis numéros de série Excel) -> on les unifie.
parse_month <- function(v) {
  if (grepl("/", v)) {
    p <- as.integer(trimws(strsplit(v, "/")[[1]]))        # c(mois, année)
    as.Date(sprintf("%04d-%02d-01", p[2], p[1]))
  } else {
    floor_date(as.Date(as.numeric(v), origin = "1899-12-30"), "month")
  }
}
prod$month <- as.Date(vapply(prod$date, function(v) as.character(parse_month(v)), ""))
num <- function(x) suppressWarnings(as.numeric(x))
prod <- prod |>
  mutate(across(c(steg, ipp, ipp_solar, auto), num)) |>
  mutate(ipp_solar = coalesce(ipp_solar, 0), auto = coalesce(auto, 0),   # absents avant 2023
         production_gwh = steg + ipp + ipp_solar + auto,                 # total RECALCULÉ (la colonne source contient des formules)
         core = steg + ipp) |>                                            # cible : définition constante (IPP intégré à la STEG dès 06/2022)
  select(month, steg, ipp, ipp_solar, auto, production_gwh, core)

# Température : journalier (Open-Meteo, réanalyse, point de grille Tunis) -> agrégats mensuels
w <- read.csv("data/open-meteo-36.80N10.17E14m.csv", skip = 3, col.names = c("date", "tmean", "tmax", "tmin"))
w$month <- floor_date(as.Date(w$date), "month")
temp <- w |> group_by(month) |>
  summarise(t_mean = mean(tmean), t_max_mean = mean(tmax), t_max_peak = max(tmax), t_min_mean = mean(tmin),
            days_above_35 = sum(tmax > 35),                    # jours de forte chaleur
            cdd24 = sum(pmax(tmean - 24, 0)), .groups = "drop")  # degrés-jours de climatisation (> 24 °C)

df <- inner_join(prod, temp, by = "month") |> arrange(month) |>
  mutate(covid_2020_04 = as.integer(month == as.Date("2020-04-01")))     # indicatrice justifiée en 2.5
stopifnot(!anyNA(df), all(diff(df$month) %in% 28:31))                    # aucun mois manquant
cat(sprintf("%d mois, %s -> %s\n", nrow(df), format(min(df$month)), format(max(df$month))))
write.csv(mutate(df, across(where(is.numeric), \(x) round(x, 3))), "data/monthly_dataset_R.csv", row.names = FALSE)

# Découpage CHRONOLOGIQUE 90/10 (jamais aléatoire). Tout le reste de l'EDA utilise le train seulement.
n_train <- floor(0.9 * nrow(df)); train <- df[1:n_train, ]; test <- df[(n_train + 1):nrow(df), ]
cat(sprintf("train : %s -> %s (%d mois) | test : %s -> %s (%d mois)\n",
            format(min(train$month), "%Y-%m"), format(max(train$month), "%Y-%m"), nrow(train),
            format(min(test$month), "%Y-%m"), format(max(test$month), "%Y-%m"), nrow(test)))

# Objet série temporelle natif avec fréquence déclarée (12 = mensuel)
y  <- ts(train$core, start = c(2005, 1), frequency = 12)
ly <- log(y)                                   # log : saisonnalité multiplicative (voir 2.1)

# ---------------------------------------------------------------- 2.1 VISUALISATION -------
png("figures_R/01_series.png", 1100, 700, res = 110)
par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
plot(ts(df$core, start = c(2005, 1), frequency = 12), ylab = "GWh / mois", main = "Production mensuelle (STEG + IPP)")
abline(v = decimal_date(min(test$month)), col = "orange", lwd = 2)
plot(ts(df$t_max_mean, start = c(2005, 1), frequency = 12), col = "red", ylab = "°C", main = "Température max moyenne, Tunis")
dev.off()

# Amplitude saisonnière annuelle brute vs log : justifie le passage au log
yr <- year(train$month)
rng_raw <- tapply(train$core, yr, function(v) max(v) - min(v))
rng_log <- tapply(log(train$core), yr, function(v) max(v) - min(v))
lvl     <- tapply(train$core, yr, mean)
cat("corr(amplitude brute, niveau annuel) =", round(cor(rng_raw, lvl), 2), "\n")
cat("coef. de variation de l'amplitude : brute =", round(sd(rng_raw) / mean(rng_raw), 2),
    "| log =", round(sd(rng_log) / mean(rng_log), 2), "\n")

# Relation production / chaleur : brute (surtout de la saisonnalité commune) puis sans saisonnalité
cat("corr. brutes (niveaux) :\n"); print(round(cor(train[, c("core", "t_mean", "t_max_mean", "days_above_35", "cdd24")])[1, ], 2))
cat("corr. des différences à 12 mois (t_mean) :", round(cor(diff(train$core, 12), diff(train$t_mean, 12)), 2), "\n")

# ---------------------------------------------------------------- 2.2 DÉCOMPOSITION -------
# (a) classique additive vs multiplicative sur la série brute : lequel laisse les résidus les plus homogènes ?
dec_add <- decompose(y, "additive"); dec_mul <- decompose(y, "multiplicative")
cat("écart-type relatif des résidus  additif :", round(sd(dec_add$random / y, na.rm = TRUE), 3),
    "| multiplicatif :", round(sd(dec_mul$random - 1, na.rm = TRUE), 3), "\n")
# (b) STL robuste sur log(core). s.window = 7 reproduit le réglage par défaut de statsmodels (Python).
fit <- stl(ly, s.window = 7, robust = TRUE)
png("figures_R/02_stl.png", 1000, 750, res = 110); plot(fit, main = "STL robuste — log(core)"); dev.off()
sea <- fit$time.series[, "seasonal"]; trd <- fit$time.series[, "trend"]; rem <- fit$time.series[, "remainder"]
cat("effet saisonnier moyen (log) par mois :\n"); print(round(tapply(sea, cycle(sea), mean), 3))
Fs <- max(0, 1 - var(rem) / var(rem + sea)); Ft <- max(0, 1 - var(rem) / var(rem + trd))
cat(sprintf("force saisonnalité = %.2f | force tendance = %.2f | écart-type du résidu (log) = %.3f\n", Fs, Ft, sd(rem)))
cat("Jarque-Bera sur le résidu STL : "); print(jarque.bera.test(as.numeric(rem)))

# ---------------------------------------------------------------- 2.3 ACF / PACF ----------
d1    <- diff(ly)                              # différence régulière (enlève la tendance)
d1D12 <- diff(diff(ly), lag = 12)              # + différence saisonnière (enlève le profil annuel)
png("figures_R/03_acf_pacf.png", 1100, 1000, res = 110)
par(mfrow = c(3, 2), mar = c(4, 4, 3, 1))
for (nm in c("log(core) brut", "Δ log(core)", "Δ Δ12 log(core)")) {
  s <- list("log(core) brut" = ly, "Δ log(core)" = d1, "Δ Δ12 log(core)" = d1D12)[[nm]]
  acf(s, lag.max = 36, main = paste("ACF —", nm)); pacf(s, lag.max = 36, main = paste("PACF —", nm))
}
dev.off()
for (nm in c("Δ log", "Δ Δ12 log")) {
  s <- if (nm == "Δ log") d1 else d1D12
  a <- acf(s, lag.max = 24, plot = FALSE)$acf[, 1, 1]; p <- pacf(s, lag.max = 24, plot = FALSE)$acf[, 1, 1]
  cat(sprintf("%-10s ACF lag1/12/24 = %.2f / %.2f / %s | PACF lag1/12 = %.2f / %.2f\n", nm, a[2], a[13],
              ifelse(is.na(a[25]), "NA", sprintf("%.2f", a[25])), p[1], p[12]))
}

# ---------------------------------------------------------------- 2.4 STATIONNARITÉ --------
# ADF (H0 : racine unitaire) ET KPSS (H0 : stationnaire) : hypothèses nulles opposées -> complémentaires.
# ADF via urca::ur.df avec choix du retard par AIC (comme autolag='AIC' en Python) ; on compare la
# statistique à la valeur critique à 5 %. KPSS via tseries::kpss.test (p-value interpolée dans [0,01 ; 0,1]).
stationarity <- function(s, name, trend = FALSE) {
  s <- as.numeric(na.omit(s))
  a  <- ur.df(s, type = if (trend) "trend" else "drift", lags = 14, selectlags = "AIC")
  st <- a@teststat[1, if (trend) "tau3" else "tau2"]
  cv <- a@cval[if (trend) "tau3" else "tau2", "5pct"]
  k  <- suppressWarnings(kpss.test(s, null = if (trend) "Trend" else "Level", lshort = FALSE))
  adf_rej <- st < cv; kpss_rej <- k$p.value < .05
  verdict <- if (adf_rej && !kpss_rej) "stationnaire" else if (!adf_rej && kpss_rej) "NON stationnaire" else
             if (adf_rej && kpss_rej) "contradictoire (ADF et KPSS rejettent)" else "non concluant (aucun ne rejette)"
  data.frame(serie = name, spec = if (trend) "const.+tend." else "const.", ADF_stat = round(st, 2),
             ADF_crit5 = round(cv, 2), KPSS_stat = round(unname(k$statistic), 2), KPSS_p = k$p.value, verdict = verdict)
}
tab <- rbind(
  stationarity(y, "core (brut)"), stationarity(y, "core (brut)", TRUE),
  stationarity(ly, "log core"),   stationarity(ly, "log core", TRUE),
  stationarity(d1, "dlog core"), stationarity(diff(ly, 12), "D12 log core"), stationarity(d1D12, "dD12 log core"),
  stationarity(train$t_mean, "t_mean"),         stationarity(train$t_mean, "t_mean", TRUE),
  stationarity(train$t_max_mean, "t_max_mean"), stationarity(train$t_max_mean, "t_max_mean", TRUE))
print(tab, row.names = FALSE)

# ---------------------------------------------------------------- 2.5 VALEURS ABERRANTES ---
r <- as.numeric(rem); iqr <- IQR(r); q <- quantile(r, c(.25, .75))
flag <- r < q[1] - 1.5 * iqr | r > q[2] + 1.5 * iqr
z <- (r - median(r)) / (1.4826 * median(abs(r - median(r))))            # z robuste (MAD), insensible aux queues épaisses
cat(sprintf("mois hors IQR : %d / %d (%.0f%%) | |z robuste| > 3,5 : %d\n", sum(flag), length(r), 100 * mean(flag), sum(abs(z) > 3.5)))
clim <- ave(train$t_max_mean, month(train$month))                        # normale mensuelle de t_max
top <- order(-abs(z))[1:8]
print(data.frame(mois = format(train$month[top], "%Y-%m"), residu = round(r[top], 3), z_robuste = round(z[top], 1),
                 anomalie_tmax = round((train$t_max_mean - clim)[top], 1), jours_sup35 = train$days_above_35[top]), row.names = FALSE)
for (v in c("t_mean", "t_max_mean", "days_above_35", "cdd24")) {
  an <- train[[v]] - ave(train[[v]], month(train$month))
  cat(sprintf("corr(résidu STL, anomalie %-14s) = %.2f\n", v, cor(r, an)))
}
cat("\nDécisions : cible = log(core) ; d = 1, D = 1, s = 12 ; départ SARIMA(0,1,1)(0,1,1)[12] ; indicatrice avril 2020.\n")