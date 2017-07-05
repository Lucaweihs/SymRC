#' Independence Testing Using Symmetric Rank Covariances.
#'
#' Tests the null hypothesis that two random vectors X and Y are independent
#' using symmetric rank covariances (SRCs). In particular, a permutation test is
#' performed using a U-statistic estimating a specified SRC.
#'
#' @export
#'
#' @inheritParams pTStar
#' @param measure a string representing the SRC to use, can be one of
#' \itemize{
#'  \item{\code{\link{pTStar}}}
#'  \item{\code{\link{jTStar}}}
#'  \item{\code{\link{hoeffD}}}
#'  \item{\code{\link{hoeffR}}}
#' }
#' @param resamples the number of times permutation test iterations to use,
#'                  more iterations means a more stable p-value but long
#'                  computation time.
#' @return an "SymRCTest" object (i.e. a list) recording the results of the
#'         independence test.
#'
#' @examples
#' # Test whether X = (x1, x2) is independent from Y = (x1 * x2)^2 + (noise)
#' # Using the hoeffD measure.
#' set.seed(2)
#' n = 150
#' x1 = rnorm(n)
#' x2 = rnorm(n)
#' X = cbind(x1, x2)
#' Y = matrix((x1 * x2)^2 + rnorm(n))
#' symRCTest(X, Y, measure = "hoeffD")
symRCTest <- function(X, Y, measure = "hoeffD", resamples = 1000) {
  if (length(resamples) != 1 || resamples %% 1 != 0) {
    stop("resamples must be integer valued.")
  }

  if (measure == "hoeffD") {
    func = hoeffD
  } else if (measure == "hoeffR") {
    func = hoeffR
  } else if (measure == "pTStar") {
    func = pTStar
  } else if (measure == "jTStar") {
    func = jTStar
  } else {
    stop(paste("Measure", measure, "is not defined."))
  }

  toReturn = list()
  class(toReturn) = "SymRCTest"
  toReturn$measure = measure
  toReturn$X = X
  toReturn$Y = Y
  toReturn$val = func(X, Y)
  toReturn$resamples = resamples
  toReturn$call = match.call()

  samples = numeric(resamples)
  for (i in 1:resamples) {
    samples[i] = func(X, Y[sample(1:nrow(Y)),])
  }
  toReturn$pVal = mean(samples >= toReturn$val)
  return(toReturn)
}

#' Print Test Results
#'
#' A simple print function for SymRCTest objects.
#'
#' @export
#'
#' @param x the SymRCTest object to be printed
#' @param ... ignored.
print.SymRCTest <- function(x, ...) {
  cat(paste("Call: ", deparse(x$call), sep = ""), "\n\n")
  cat(paste("Permutation test (", x$resamples," simulations)\n", sep = ""))
  cat(paste("Number of samples:", nrow(x$X), "\n\n"))

  cat(paste("Results:\n"))
  df = data.frame(round(x$val, 5), round(x$pVal, 5))
  colnames(df) = c(paste(x$measure, "value"), "Perm. p-val")
  row.names(df) = ""
  print(df)
}