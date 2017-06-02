#' Estimate the Multivariate Partial Tau Star Measure
#'
#' Computes, from data, the U-statistic estimating the multivariate partial tau
#' star measure. This statistic estimates the dependence between two collections
#' of random variables X and Y.
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
#'
#' @return the U-statistic for the data
pTStar <- function(X, Y, method = "auto") {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 4)

  if (method == "auto") {
    if (ncol(X) == ncol(Y) && ncol(X) == 1) {
      return(TauStar::tStar(X, Y))
    }
    method = "standard"
  }

  if (method == "naive") {
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
#' of random variables X and Y.
#'
#' @export
#'
#' @inheritParams pTStar
#'
#' @return the U-statistic for the data
jTStar <- function(X, Y, method = "auto") {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 4)

  if (method == "auto") {
    if (ncol(X) == ncol(Y) && ncol(X) == 1) {
      return(TauStar::tStar(X, Y))
    }
    method = "naive"
  }

  if (method == "naive") {
    return(jointTauStarFromDef(X, Y, rep(1, ncol(X)), rep(1, ncol(Y))))
  } else if (method == "range-tree") {
    return(jointTauStarRangeTree(X, Y, rep(1, ncol(X)), rep(1, ncol(Y))))
  } else {
    stop("Invalid choice of method in jTStar.")
  }
}
