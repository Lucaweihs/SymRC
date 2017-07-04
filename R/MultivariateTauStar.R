#' Estimate the Multivariate Partial Tau Star Measure
#'
#' Computes, from data, the U-statistic estimating the multivariate partial tau
#' star measure. This statistic estimates the dependence between two collections
#' of random variables X and Y. This U-statistic takes on
#' values between 0 and 2/3 when X and Y are 1-dimensional and between 0 and 2
#' when either X or Y are more than 1-dimensional.
#'
#' @export
#'
#' @param X an n-by-l matrx where n is the sample size and l is the number of
#'          covariates in the first group.
#' @param Y an n-by-m matrx where n is the sample size and m is the number of
#'          covariates in the second group.
#' @param method A string indicating which algorithm to use to compute the
#'        U-statistic. Currently there are three options
#'\itemize{
#'  \item{\code{"auto"} - Function attemps to automatically pick the fastest
#'  method for the given data.}
#'  \item{\code{"def"} - Computes the U-statistic by definition, takes
#'   O(n^4) time.}
#'  \item{\code{"range-tree"} - A more complicated algorithm which takes
#'  O(n^2*log(n)^(2*(l+m)) time. While this is asymptotically faster than
#'  the naive algorithm it, in practice, can be slower due to
#'  large constant factors hidden by the asymptotic analysis.}
#' }
#' This argument is ignored if approx is TRUE.
#' @param approx if \code{approx} is \code{TRUE} then instead of computing the
#'               u-statistic using all of the data we instead approximate it by
#'               random sampling, see \code{approxControl} for how
#'               to set the number of samples used.
#' @param approxControl a named list, should be of the form
#'                     \code{list(nsims = NSIMS)} if you would like to
#'                     approximate the u-statistic using \code{NSIMS} samples,
#'                     if instead you would like use as many samples as possible
#'                     within approximately some number of seconds, say
#'                     \code{NSECONDS}, then use a list of the form
#'                     \code{list(nseconds = NSECONDS)}.
#'
#' @return the (estimated if approx != 0) U-statistic for the data
#'
#' @examples
#' # Bivariate dependence
#' set.seed(1)
#' x = rnorm(30)
#' y = x^2 + rnorm(30)
#' pTStar(x, y)
#'
#' # Trivariate dependence
#' x1 = rnorm(30)
#' x2 = rnorm(30)
#' y = (x1 * x2)^2 + rnorm(30)
#' pTStar(cbind(x1, x2), y)
#'
#' # Approximate the above as well as we can in 5 seconds
#' pTStar(cbind(x1, x2), y, approx = T, approxControl = list(nseconds = 5))
pTStar <- function(X, Y, method = "auto", approx = FALSE,
                   approxControl = list(nreps = 10^4)) {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 4)

  if (approx != 0) {
    if (is.null(approxControl$nreps)) {
      approxControl$nreps = 0
    }
    if (is.null(approxControl$nseconds)) {
      approxControl$nseconds = 0
    }
    return(partialTauStarApprox(X, Y, approxControl$nreps, approxControl$nseconds))
  }

  if (method == "auto") {
    if (ncol(X) == ncol(Y) && ncol(X) == 1) {
      return(TauStar::tStar(X, Y))
    }
    method = "def"
  }

  if (method == "def") {
    return(partialTauStarFromDef(X, Y))
  } else if (method == "range-tree") {
    return(partialTauStarRangeTree(X, Y))
  } else {
    stop("Invalid choice of method in pTStar.")
  }
}

#' Estimate the Multivariate Joint Tau Star Measure
#'
#' Computes, from data, the U-statistic estimating the multivariate joint tau
#' star measure. This statistic estimates the dependence between two collections
#' of random variables X and Y. This U-statistic takes on
#' values between 0 and 2/3.
#'
#' @export
#'
#' @inheritParams pTStar
#'
#' @return the U-statistic for the data
#'
#' @examples
#' # Bivariate dependence
#' set.seed(1)
#' x = rnorm(30)
#' y = x^2 + rnorm(30)
#' jTStar(x, y)
#'
#' # Trivariate dependence
#' x1 = rnorm(30)
#' x2 = rnorm(30)
#' y = (x1 * x2)^2 + rnorm(30)
#' jTStar(cbind(x1, x2), y)
#'
#' # Approximate the above as well as well as we can in 5 seconds
#' jTStar(cbind(x1, x2), y, approx = T, approxControl = list(nseconds = 5))
jTStar <- function(X, Y, method = "auto", approx = FALSE,
                   approxControl = list(nreps = 10^4)) {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 4)

  if (approx != 0) {
    if (is.null(approxControl$nreps)) {
      approxControl$nreps = 0
    }
    if (is.null(approxControl$nseconds)) {
      approxControl$nseconds = 0
    }
    return(jointTauStarApprox(X, Y, rep(1, ncol(X)), rep(1, ncol(Y)),
                              approxControl$nreps, approxControl$nseconds))
  }

  if (method == "auto") {
    d = ncol(X) + ncol(Y)
    if (d == 2) {
      return(TauStar::tStar(X, Y))
    }

    if (nrow(X) <= 10) {
      method = "def"
    } else {
      method = "range-tree"
    }
  }

  if (method == "def") {
    return(jointTauStarFromDef(X, Y, rep(1, ncol(X)), rep(1, ncol(Y))))
  } else if (method == "range-tree") {
    return(jointTauStarRangeTree(X, Y, rep(1, ncol(X)), rep(1, ncol(Y))))
  } else {
    stop("Invalid choice of method in jTStar.")
  }
}
