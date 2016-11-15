library(WCM)
context("Testing the naive U-statistic functions.")

kendallsTauTies <- function(x, y) {
  val = 0;
  for (i in 1:(length(x) - 1)) {
    for (j in (i + 1):length(x)) {
      val = val + sign((x[i] - x[j]) * (y[i] - y[j]))
    }
  }
  return(val / choose(length(x), 2))
}

spearmansRhoTies <- function(x, y) {
  val = 0;
  for (i in 1:length(x)) {
    for (j in 1:length(x)) {
      for (k in 1:length(x)) {
        if (i != j && i != k && j != k) {
          val = val + 3 * sign((x[i] - x[j]) * (y[i] - y[k]))
        }
      }
    }
  }
  return(val / (6 * choose(length(x), 3)))
}

test_that("Naive kendall's tau statistic works", {
  set.seed(123)
  for (i in 1:10) {
    X = matrix(rnorm(15), ncol = 1)
    Y = matrix(rnorm(15), ncol = 1)

    a = kendallsTauNaive(X, Y)
    b = kendallsTauTies(X, Y)
    c = kendallsTauNaiveApprox(X, Y, 100000)

    expect_equal(a, b)
    expect_true(abs(a - c) < .02)
  }

  for (i in 1:10) {
    X = matrix(rpois(15, lambda = 5), ncol = 1)
    Y = matrix(rpois(15, lambda = 5), ncol = 1)

    a = kendallsTauNaive(X, Y)
    b = kendallsTauTies(X, Y)
    c = kendallsTauNaiveApprox(X, Y, 100000)

    expect_equal(a, b)
    expect_true(abs(a - c) < .02)
  }

  for (i in 1:10) {
    X = matrix(rnorm(15), ncol = 1)
    Y = matrix(rnorm(15), ncol = 1)

    a = spearmansRhoNaive(X,Y)
    b = spearmansRhoTies(X,Y)
    c = spearmansRhoNaiveApprox(X,Y, 100000)

    expect_equal(a, b)
    expect_true(abs(a - c) < .02)
  }

})

