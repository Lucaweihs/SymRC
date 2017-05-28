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
