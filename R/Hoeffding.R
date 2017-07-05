#' Efficiently Compute the Multivariate Hoeffding's D Statistic
#'
#' Computes, from data, the U-statistic estimating Hoeffding's D, a measure
#' of dependence between random vectors X and Y. This U-statistic takes on
#' values between 0 and 1/120.
#'
#' @export
#'
#' @inheritParams pTStar
#' @param method A string indicating which algorithm to use to compute the
#'        U-statistic. Currently there are three options
#'\itemize{
#'  \item{\code{"auto"} - Function attemps to automatically pick the fastest
#'  method for the given data.}
#'  \item{\code{"def"} - Computes the U-statistic by definition, takes
#'   O(n^5) time.}
#'  \item{\code{"standard"} - A better implementation which takes O(n^2) time.}
#'  \item{\code{"range-tree"} - A version of the \code{"standard"} algorithm
#'  which uses a range tree data structure to do orthogonal range queries. This
#'  version of the algorithm takes O(n*log(n)^(l+m)) time.}
#' }
#'
#' @return the U-statistic for the data
#'
#' @examples
#' # Bivariate dependence
#' set.seed(1)
#' x = rnorm(1000)
#' y = x^2 + rnorm(1000)
#' hoeffD(x, y)
#'
#' # Trivariate dependence
#' x1 = rnorm(1000)
#' x2 = rnorm(1000)
#' y = (x1 * x2)^2 + rnorm(1000)
#' hoeffD(cbind(x1, x2), y)
hoeffD <- function(X, Y, standardize = FALSE, method = "auto") {
  if (standardize) {
    a = hoeffD(X, Y, standardize = F, method = method)
    b = hoeffD(X, X, standardize = F, method = method)
    c = hoeffD(Y, Y, standardize = F, method = method)
    if (b == 0 || c == 0) {
      warning(paste("denominator in standardization is 0."))
      return(NA)
    }
    return(a / sqrt(b * c))
  }
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 5)

  d = ncol(X) + ncol(Y)
  n = nrow(X)
  if (method == "auto") {
    method = "range-tree"
    if ((d == 2 && n < 400) ||
        (d == 3 && n < 3500) ||
        (d > 3 && d > 1 + log(n / 9, base = log2(n)))) {
      method = "standard"
    }
  }

  if (method == "def") {
    return(hoeffdingDFromDef(X, Y))
  } else if (method == "standard") {
    return(hoeffdingDNaive(X, Y))
  } else if (method == "range-tree") {
    return(hoeffdingDRangeTree(X, Y))
  } else {
    stop("Invalid choice of method in hoeffD.")
  }
}

#' Efficiently Compute the Multivariate Hoeffding's R Statistic
#'
#' Computes, from data, the U-statistic estimating Hoeffding's R, a measure
#' of dependence between random vectors X and Y. This U-statistic takes on
#' values between 0 and 1/120.
#'
#' @export
#'
#' @inheritParams pTStar
#' @param method A string indicating which algorithm to use to compute the
#'        U-statistic. Currently there are three options
#'\itemize{
#'  \item{\code{"auto"} - Function attemps to automatically pick the fastest
#'  method for the given data.}
#'  \item{\code{"def"} - Computes the U-statistic by definition, takes
#'   O(n^(4 + l + m)) time.}
#'  \item{\code{"standard"} - A better implementation which takes
#'  O(n^(1 + l + m)) time.}
#'  \item{\code{"orth"} - A version of the \code{"standard"} algorithm
#'  which uses a orthogonal range tensor data structure to do orthogonal
#'  range queries. This version of the algorithm takes O(n^(l + m)) time.}
#' }
#'
#' @return the U-statistic for the data
#'
#' @examples
#' # Bivariate dependence
#' set.seed(1)
#' x = rnorm(100)
#' y = x^2 + rnorm(100)
#' hoeffR(x, y)
#'
#' # Trivariate dependence
#' x1 = rnorm(100)
#' x2 = rnorm(100)
#' y = (x1 * x2)^2 + rnorm(100)
#' hoeffR(cbind(x1, x2), y)
hoeffR <- function(X, Y, standardize = FALSE, method = "auto") {
  if (standardize) {
    a = hoeffR(X, Y, standardize = F, method = method)
    b = hoeffR(X, X, standardize = F, method = method)
    c = hoeffR(Y, Y, standardize = F, method = method)
    if (b == 0 || c == 0) {
      warning(paste("denominator in standardization is 0."))
      return(NA)
    }
    return(a / sqrt(b * c))
  }
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, ncol(X) + ncol(Y) + 4)

  if (method == "auto") {
    method = "orth"
  }

  if (method == "def") {
    return(hoeffdingRFromDef(X, Y))
  } else if (method == "standard") {
    return(hoeffdingRNaive(X, Y))
  } else if (method == "orth") {
    return(hoeffdingROrthTensor(X, Y))
  } else {
    stop("Invalid choice of method in hoeffD.")
  }
}
