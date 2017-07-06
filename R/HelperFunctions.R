checkMatrices <- function(X, Y, order) {
  if (ncol(X) == 0 || ncol(Y) == 0) {
    stop("Matrices must have more than 0 columns.")
  }
  if (nrow(X) != nrow(Y)) {
    stop("X and Y matrices must have the same number of rows.")
  }
  if (nrow(X) < order) {
    stop(paste("Input matrices X and Y must have at least", order, "rows."))
  }
}

getCurrentSeed <- function() {
  if (exists(".Random.seed", .GlobalEnv)) {
    seed <- .GlobalEnv$.Random.seed
  } else {
    seed <- NULL
  }
  return(seed)
}

restoreSeed <- function(seed) {
  if (!is.null(seed)) {
    .GlobalEnv$.Random.seed <- seed
  } else if (exists(".Random.seed", .GlobalEnv)) {
    rm(".Random.seed", envir = .GlobalEnv)
  }
}

computeStandardizedStat <- function(func, X, Y) {
  seed = getCurrentSeed()
  a = func(X, Y)
  restoreSeed(seed)
  b = func(X, X)
  restoreSeed(seed)
  c = func(Y, Y)
  if (b == 0 || c == 0) {
    warning(paste("denominator in standardization is 0."))
    return(NA)
  }
  return(a / sqrt(b * c))
}
