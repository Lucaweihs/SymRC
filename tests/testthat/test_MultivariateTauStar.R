library(WCM)
library(TauStar)
context("Testing the multivariate tau star measures.")

test_that("Check that multivariate measures agree with t* in 2 dims", {

  set.seed(123)
  for (i in 1:10) {
    X = matrix(rnorm(30), ncol = 1)
    Y = matrix(rnorm(30), ncol = 1)
    a = tStar(X, Y)
    b = 4 * partialTauStarNaive(X, Y)
    c = 4 * lexTauStarNaive(X, Y, 0, 0)
    d = 4 * fullLexTauStarNaive(X, Y)

    expect_equal(a, b)
    expect_equal(a, c)
    expect_equal(a, d)
  }

  for (i in 1:10) {
    X = matrix(rpois(30, lambda = 2), ncol = 1)
    Y = matrix(rpois(30, lambda = 2), ncol = 1)
    a = tStar(X, Y)
    b = 4 * partialTauStarNaive(X, Y)
    c = 4 * lexTauStarNaive(X, Y, 0, 0)
    d = 4 * fullLexTauStarNaive(X, Y)

    expect_equal(a, b)
    expect_equal(a, c)
    expect_equal(a, d)
  }
})


test_that("Check that the RangeTree version of partial tau* agrees with naive", {
  n = 15
  nXCols = 1
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    a = partialTauStarNaive(X, Y)
    b = partialTauStar(X, Y)
    expect_equal(a, b)
  }

  n = 15
  nXCols = 2
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    a = partialTauStarNaive(X, Y)
    b = partialTauStar(X, Y)
    expect_equal(a, b)
  }

  n = 15
  nXCols = 1
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    a = partialTauStarNaive(X, Y)
    b = partialTauStar(X, Y)
    expect_equal(a, b)
  }

  n = 10
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    a = partialTauStarNaive(X, Y)
    b = partialTauStar(X, Y)
    expect_equal(a, b)
  }

  n = 6
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rnorm(nXCols * n), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    a = partialTauStarNaive(X, Y)
    b = partialTauStar(X, Y)
    expect_equal(a, b)
  }
})

# library(foreach)
# library(doRNG)
# library(doParallel)
# registerDoParallel(7)
#
# set.seed(12)
# n = 1000
# X = matrix(rnorm(2 * n), ncol = 2)
# Y = matrix(rnorm(n), ncol = 1)
# Y = Y + X[,1]*X[,2]
# tauStarTest(X[,1], Y)
#
# a = partialTauStarNaive(X, Y)
# b = lexTauStarNaive(X, Y, c(0,1), 0)
# c = fullLexTauStarNaive(X, Y)
#
# sims = 100
#
# permValues = foreach(i = 1:sims, .combine = 'rbind') %dorng% {
#   if (i %% 10 == 0) print(i)
#   Xpermed = X[sample(1:n),]
#   Ypermed = Y[sample(1:n),, drop = F]
#   c(partialTauStarNaive(Xpermed, Ypermed),
#     lexTauStarNaive(Xpermed, Ypermed, c(0,1), 0),
#     fullLexTauStarNaive(Xpermed, Ypermed))
# }
#
# mean(permValues[,1] < a)
# mean(permValues[,2] < b)
# mean(permValues[,3] < c)
#
#
