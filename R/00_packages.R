# Installe les packages R nécessaires (à lancer une fois, sur une machine avec accès à CRAN).
pkgs <- c("readxl", "lubridate", "dplyr", "forecast", "tseries", "urca", "zoo", "vars", "strucchange", "lmtest", "rugarch", "FinTS")
miss <- setdiff(pkgs, rownames(installed.packages()))
if (length(miss)) install.packages(miss)
invisible(lapply(pkgs, library, character.only = TRUE))
sessionInfo()
