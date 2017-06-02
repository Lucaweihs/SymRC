library(SymRC)
source("TestHelpers.R")
context("Testing Hoeffding's D and R measures.")

test_that("Check that hoeffding's D agrees with the naive version", {
  set.seed(123)
  n = 10
  nXCols = 1
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rnorm(nXCols * n), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    expect_all_equal(hoeffdingDFromDef(X, Y),
                     hoeffdingDNaive(X, Y),
                     hoeffdingDRangeTree(X, Y))
  }

  n = 15
  nXCols = 2
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingDFromDef(X, Y),
                     hoeffdingDNaive(X, Y),
                     hoeffdingDRangeTree(X, Y))
  }

  n = 15
  nXCols = 1
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingDFromDef(X, Y),
                     hoeffdingDNaive(X, Y),
                     hoeffdingDRangeTree(X, Y))
  }

  n = 20
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingDFromDef(X, Y),
                     hoeffdingDNaive(X, Y),
                     hoeffdingDRangeTree(X, Y))
  }

  n = 20
  nXCols = 4
  nYCols = 4
  for (i in 1:10) {
    X = matrix(rnorm(nXCols * n), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    expect_all_equal(hoeffdingDFromDef(X, Y),
                     hoeffdingDNaive(X, Y),
                     hoeffdingDRangeTree(X, Y))
  }
})

test_that("Check that hoeffding's R agrees with the naive version", {
  set.seed(123)
  n = 10
  nXCols = 1
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingRFromDef(X, Y),
                     hoeffdingRNaive(X, Y),
                     hoeffdingROrthTensor(X, Y))
  }

  n = 10
  nXCols = 2
  nYCols = 1
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingRFromDef(X, Y),
                     hoeffdingRNaive(X, Y),
                     hoeffdingROrthTensor(X, Y))
  }

  n = 10
  nXCols = 1
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rpois(nYCols * n, 1), ncol = nYCols)
    expect_all_equal(hoeffdingRFromDef(X, Y),
                     hoeffdingRNaive(X, Y),
                     hoeffdingROrthTensor(X, Y))
  }

  n = 10
  nXCols = 2
  nYCols = 2
  for (i in 1:10) {
    X = matrix(rpois(nXCols * n, 1), ncol = nXCols)
    Y = matrix(rnorm(nYCols * n), ncol = nYCols)
    expect_all_equal(hoeffdingRFromDef(X, Y),
                     hoeffdingRNaive(X, Y),
                     hoeffdingROrthTensor(X, Y))
  }
})
