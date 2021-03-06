% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Hoeffding.R
\name{hoeffR}
\alias{hoeffR}
\title{Efficiently Compute the Multivariate Hoeffding's R Statistic}
\usage{
hoeffR(X, Y, standardize = FALSE, method = "auto")
}
\arguments{
\item{X}{an n-by-l matrx where n is the sample size and l is the number of
covariates in the first group.}

\item{Y}{an n-by-m matrx where n is the sample size and m is the number of
covariates in the second group.}

\item{standardize}{if TRUE then this function returns a estimate of the
"standardized" version of the statistic which is bounded between -1
and 1. That is, letting the multivariate measure be denoted m, then
when \code{standardize = TRUE} this function returns an estimate of
m(X,Y) / sqrt(m(X,X) * m(Y,Y)). Estimates of m(X,X) and m(X,Y) are
done using settings defined below, if either m(X,X) or m(Y,Y) equal
0 then NA is returned and a warning is printed.}

\item{method}{A string indicating which algorithm to use to compute the
       U-statistic. Currently there are three options
\itemize{
 \item{\code{"auto"} - Function attemps to automatically pick the fastest
 method for the given data.}
 \item{\code{"def"} - Computes the U-statistic by definition, takes
  O(n^(4 + l + m)) time.}
 \item{\code{"standard"} - A better implementation which takes
 O(n^(1 + l + m)) time.}
 \item{\code{"orth"} - A version of the \code{"standard"} algorithm
 which uses a orthogonal range tensor data structure to do orthogonal
 range queries. This version of the algorithm takes O(n^(l + m)) time.}
}}
}
\value{
the U-statistic for the data
}
\description{
Computes, from data, the U-statistic estimating Hoeffding's R, a measure
of dependence between random vectors X and Y. This U-statistic takes on
values between 0 and 1/120.
}
\examples{
# Bivariate dependence
set.seed(1)
x = rnorm(100)
y = x^2 + rnorm(100)
hoeffR(x, y)

# Trivariate dependence
x1 = rnorm(100)
x2 = rnorm(100)
y = (x1 * x2)^2 + rnorm(100)
hoeffR(cbind(x1, x2), y)
}
