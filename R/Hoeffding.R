#' Efficiently Compute the Multivariate Hoeffding's D Statistic
#'
#' Computes, from data, the U-statistic estimating Hoeffding's D, a measure
#' of dependence between random vectors X and Y.
#'
#' @export
#'
#' @inheritParams pTStar
#' @param method A string indicating which algorithm to use to compute the
#'        U-statistic. Currently there are three options
#'\itemize{
#'  \item{\code{"auto"} - Function attemps to automatically pick the fastest
#'  method for the given data.}
#'  \item{\code{"naive"} - A naive implementation which takes O(n^5) time.}
#'  \item{\code{"range-tree"} - A more complicated algorithm which takes
#'  O(n*log(n)^(l+m)) time. This tends to me much faster than the naive
#'  algorithm.}
#' }
#'
#' @return the U-statistic for the data
hoeffD <- function(X, Y, method = "auto") {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, 5)

  if (method == "auto") {
    method = "range-tree"
  }

  if (method == "naive") {
    return(hoeffdingDNaive(X, Y))
  } else if (method == "range-tree") {
    return(hoeffdingD(X, Y))
  } else {
    stop("Invalid choice of method in hoeffD.")
  }
}

#' Efficiently Compute the Multivariate Hoeffding's R Statistic
#'
#' Computes, from data, the U-statistic estimating Hoeffding's R, a measure
#' of dependence between random vectors X and Y.
#'
#' @export
#'
#' @inheritParams pTStar
#' @param method A string indicating which algorithm to use to compute the
#'        U-statistic. Currently there are three options
#'\itemize{
#'  \item{\code{"auto"} - Function attemps to automatically pick the fastest
#'  method for the given data.}
#'  \item{\code{"naive"} - A naive implementation which takes
#'  O(n^(4+l+m)) time.}
#'  \item{\code{"orth"} - A more complicated algorithm which takes
#'  O(n^(l+m)) time. Except in cases where l+m is large and n is small
#'  this algorithm tends to be significantly faster than the naive version.}
#' }
#'
#' @return the U-statistic for the data
hoeffR <- function(X, Y, method = "auto") {
  if (!is.matrix(X)) { X = matrix(X) }
  if (!is.matrix(Y)) { Y = matrix(Y) }
  checkMatrices(X, Y, ncol(X) + ncol(Y) + 4)

  if (method == "auto") {
    method = "orth"
  }

  if (method == "naive") {
    return(hoeffdingRNaive(X, Y))
  } else if (method == "orth") {
    return(hoeffdingR(X, Y))
  } else {
    stop("Invalid choice of method in hoeffD.")
  }
}
