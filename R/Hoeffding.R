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
#'  \item{\code{"def"} - Computes the U-statistic by definition, takes
#'   O(n^5) time.}
#'  \item{\code{"standard"} - A better implementation which takes O(n^2) time.}
#'  \item{\code{"range-tree"} - A version of the \code{"standard"} algorithm
#'  which uses a range tree data structure to do orthogonal range queries. This
#'  version of the algorithm takes O(n*log(n)^(l+m)) time.}
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
hoeffR <- function(X, Y, method = "auto") {
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
