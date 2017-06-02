library(SymRC)
library(TauStar)
source("TestHelpers.R")
context("Testing the multivariate tau star measures.")

test_that("Check that multivariate measures agree with t* in 2 dims", {

  set.seed(123)
  for (i in 1:10) {
    X = matrix(rnorm(30), ncol = 1)
    Y = matrix(rnorm(30), ncol = 1)
    a = tStar(X, Y)
    b = partialTauStarFromDef(X, Y)
    c = lexTauStarFromDef(X, Y, 0, 0)
    d = fullLexTauStarFromDef(X, Y)
    e = jointTauStarFromDef(X, Y, 1, 1)
    expect_all_equal(a, b, c, d, e)
  }

  for (i in 1:10) {
    X = matrix(rpois(30, lambda = 2), ncol = 1)
    Y = matrix(rpois(30, lambda = 2), ncol = 1)
    a = tStar(X, Y)
    b = partialTauStarFromDef(X, Y)
    c = lexTauStarFromDef(X, Y, 0, 0)
    d = fullLexTauStarFromDef(X, Y)
    e = jointTauStarFromDef(X, Y, 1, 1)
    expect_all_equal(a, b, c, d, e)
  }
})


test_that("Check that the RangeTree version of partial tau* agrees with naive", {
  set.seed(123)

  n = 15
  nXCols = 1
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(partialTauStarFromDef(X, Y),
                     partialTauStarNaive(X, Y),
                     partialTauStarRangeTree(X, Y))
  }

  n = 15
  nXCols = 2
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(partialTauStarFromDef(X, Y),
                     partialTauStarNaive(X, Y),
                     partialTauStarRangeTree(X, Y))
  }

  n = 15
  nXCols = 1
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(partialTauStarFromDef(X, Y),
                     partialTauStarNaive(X, Y),
                     partialTauStarRangeTree(X, Y))
  }

  n = 20
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(partialTauStarFromDef(X, Y),
                     partialTauStarNaive(X, Y),
                     partialTauStarRangeTree(X, Y))
  }

  n = 10
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rnorm(nXCols * n), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    expect_all_equal(partialTauStarFromDef(X, Y),
                     partialTauStarNaive(X, Y),
                     partialTauStarRangeTree(X, Y))
  }
})

test_that("Check that the RangeTree version of joint tau* agrees with naive", {
  set.seed(1234)

  n = 15
  nXCols = 1
  nYCols = 1
  for (i in 1:10) {
    xOnOffVec = sample(c(1, rbinom(nXCols - 1, size = 1, prob = 1/2)))
    yOnOffVec = sample(c(1, rbinom(nYCols - 1, size = 1, prob = 1/2)))
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(jointTauStarFromDef(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarNaive(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarRangeTree(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec))
  }

  n = 15
  nXCols = 2
  nYCols = 1
  for (i in 1:10) {
    xOnOffVec = sample(c(1, rbinom(nXCols - 1, size = 1, prob = 1/2)))
    yOnOffVec = sample(c(1, rbinom(nYCols - 1, size = 1, prob = 1/2)))
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(jointTauStarFromDef(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarNaive(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarRangeTree(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec))
  }

  n = 15
  nXCols = 1
  nYCols = 2
  for (i in 1:10) {
    xOnOffVec = sample(c(1, rbinom(nXCols - 1, size = 1, prob = 1/2)))
    yOnOffVec = sample(c(1, rbinom(nYCols - 1, size = 1, prob = 1/2)))
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(jointTauStarFromDef(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarNaive(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarRangeTree(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec))
  }

  n = 20
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    xOnOffVec = sample(c(1, rbinom(nXCols - 1, size = 1, prob = 1/2)))
    yOnOffVec = sample(c(1, rbinom(nYCols - 1, size = 1, prob = 1/2)))
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(jointTauStarFromDef(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarNaive(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarRangeTree(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec))
  }

  n = 15
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    xOnOffVec = sample(c(1, rbinom(nXCols - 1, size = 1, prob = 1/2)))
    yOnOffVec = sample(c(1, rbinom(nYCols - 1, size = 1, prob = 1/2)))
    X = matrix(rnorm(nXCols * n), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    expect_all_equal(jointTauStarFromDef(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarNaive(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec),
                     jointTauStarRangeTree(X, Y, xOnOffVec = xOnOffVec, yOnOffVec = yOnOffVec))
  }
})
