library(SymRC)
context("Testing the datastructures for orthogonal range queries work.")

makeIndicesToCheck <- function(n, dims) {
  lowerMat = matrix(numeric(length(dims) * n), ncol = length(dims))
  upperMat = matrix(numeric(length(dims) * n), ncol = length(dims))
  for (i in 1:n) {
    for (j in 1:length(dims)) {
      lowerMat[i,j] = sample(0:dims[j], 1)
      upperMat[i,j] = sample(lowerMat[i,j]:dims[j], 1)
    }
  }
  return(list(lowerMat = lowerMat, upperMat = upperMat))
}

countWithinRange <- function(jointRanks, lowerMat, upperMat) {
  a = numeric(nrow(lowerMat))
  for (i in 1:nrow(lowerMat)) {
    a[i] = sum(apply(jointRanks, 1, function(x) { all(x >= lowerMat[i,]) }) &
                 apply(jointRanks, 1, function(x) { all(x <= upperMat[i,]) }))
  }
  return(a)
}

test_that("Orthogonal range query tests", {
  set.seed(123)

  n = 20
  m = 2
  for (i in 1:10) {
    A = matrix(rnorm(n * m), ncol = m)
    B = toJointRankMatrix(A)
    dims = apply(X = B, FUN = max, MARGIN = 2)
    indsToCheck = makeIndicesToCheck(100, dims)

    a = as.numeric(orthRangeTensorCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    b = as.numeric(alignedRangeTreeCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    c = countWithinRange(B, indsToCheck$lowerMat, indsToCheck$upperMat)
    expect_equal(a,b)
    expect_equal(a,c)
  }

  n = 20
  m = 4
  for (i in 1:10) {
    A = matrix(rnorm(n * m), ncol = m)
    B = toJointRankMatrix(A)
    dims = apply(X = B, FUN = max, MARGIN = 2)
    indsToCheck = makeIndicesToCheck(100, dims)

    a = as.numeric(orthRangeTensorCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    b = as.numeric(alignedRangeTreeCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    c = countWithinRange(B, indsToCheck$lowerMat, indsToCheck$upperMat)
    expect_equal(a,b)
    expect_equal(a,c)
  }

  n = 20
  m = 4
  for (i in 1:10) {
    A = matrix(rpois(n * m, lambda = 2), ncol = m)
    B = toJointRankMatrix(A)
    dims = apply(X = B, FUN = max, MARGIN = 2)
    indsToCheck = makeIndicesToCheck(100, dims)

    a = as.numeric(orthRangeTensorCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    b = as.numeric(alignedRangeTreeCount(B, indsToCheck$lowerMat, indsToCheck$upperMat))
    c = countWithinRange(B, indsToCheck$lowerMat, indsToCheck$upperMat)
    expect_equal(a,b)
    expect_equal(a,c)
  }
})

