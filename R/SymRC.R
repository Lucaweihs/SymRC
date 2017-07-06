#' SymRC package documentation.
#'
#' Estimates various Symmetric Rank Covariances (SRCs) using U-statistics.
#' Currently this packages provides implementations for quickly computing
#' Hoeffding's R and D statistics as well as two multivariate extensions of
#' the Bergsma/Dassios Sign Covariance tau*. See the following functions for
#' details:
#' \itemize{
#'  \item{\code{\link{pTStar}}}
#'  \item{\code{\link{jTStar}}}
#'  \item{\code{\link{hoeffD}}}
#'  \item{\code{\link{hoeffR}}}
#' }
#'
#' To test whether or not two random vectors X and Y are dependent (using the
#' above U-statistics) see the function \code{\link{symRCTest}}.
#'
#' @importFrom Rcpp evalCpp
#' @importFrom TauStar tStar
#' @useDynLib SymRC
#'
#' @name SymRC
NULL
