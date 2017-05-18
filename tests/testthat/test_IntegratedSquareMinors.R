library(SymRC)
context("Testing the ISM measures.")

hoeffD <- function(x, y) {
  n = length(x)
  val = 0;
  for (i1 in 1:n) {
    for (i2 in 1:n) {
      for (i3 in 1:n) {
        for (i4 in 1:n) {
          for (i5 in 1:n) {
            if (length(unique(c(i1,i2,i3,i4,i5))) == 5) {
              if (max(x[i1], x[i2]) <= x[i5] &&
                  x[i5] < min(x[i3], x[i4])) {
                val = val + (max(y[i1], y[i2]) <= y[i5] && y[i5] < min(y[i3], y[i4]))
                val = val + (max(y[i3], y[i4]) <= y[i5] && y[i5] < min(y[i1], y[i2]))
                val = val - 2 * (max(y[i1], y[i3]) <= y[i5] && y[i5] < min(y[i2], y[i4]))
              }
            }
          }
        }
      }
    }
  }
  return(val / (choose(n, 5) * factorial(5)))
}

indicator <- function(z1, z2, z3, z4, z5, inds0, inds1) {
  inds0 = inds0 + 1
  inds1 = inds1 + 1
  return(all(pmax(z1, z2) <= z5) &&
            all(z5[inds0] < z3[inds0]) &&
            all(z3[-inds0] <= z5[-inds0]) &&
            all(z5[inds1] < z4[inds1]) &&
            all(z4[-inds1] <= z5[-inds1]))
}

ismSlowOnce <- function(X, Y, xInds0, xInds1, yInds0, yInds1) {
  if (!is.matrix(X)) {
    X = matrix(X, ncol = 1)
  }
  if (!is.matrix(Y)) {
    Y = matrix(Y, ncol = 1)
  }
  n = nrow(X)
  val = 0;
  for (i1 in 1:n) {
    for (i2 in 1:n) {
      for (i3 in 1:n) {
        for (i4 in 1:n) {
          for (i5 in 1:n) {
            if (length(unique(c(i1,i2,i3,i4,i5))) == 5) {
              x1 = X[i1,]; x2 = X[i2,]; x3 = X[i3,]; x4 = X[i4,]; x5 = X[i5,];
              y1 = Y[i1,]; y2 = Y[i2,]; y3 = Y[i3,]; y4 = Y[i4,]; y5 = Y[i5,];
              if (indicator(x1, x2, x3, x4, x5, xInds0, xInds1)) {
                a = indicator(y1, y2, y3, y4, y5, yInds0, yInds1)
                b = indicator(y3, y4, y1, y2, y5, yInds0, yInds1)
                c = indicator(y1, y3, y2, y4, y5, yInds0, yInds1)
                val = val + a
                val = val + b
                val = val - 2 * c
                # if (a || b || c) {
                #   print(c(i1, i2, i3, i4, i5))
                # }
              }
            }
          }
        }
      }
    }
  }
  return(val / (choose(n, 5) * factorial(5)))
}

ismSlow <- function(X, Y, xInds0, xInds1, yInds0, yInds1) {
  xEq = length(xInds0) == length(xInds1) && all(xInds0 == xInds1)
  yEq = length(yInds0) == length(yInds1) && any(yInds0 == yInds1)

  a = ismSlowOnce(X, Y, xInds0, xInds1, yInds0, yInds1)
  val = a
  # if (a != 0) { print("a"); print(a) }
  if (!xEq) {
    b = ismSlowOnce(X, Y, xInds1, xInds0, yInds0, yInds1)
    val = val + b
    # if (b != 0) { print("b"); print(b) }
  }
  if (!yEq) {
    c = ismSlowOnce(X, Y, xInds0, xInds1, yInds1, yInds0)
    val = val + c
    # if (c != 0) { print("c"); print(c) }
  }
  if (!xEq && !yEq) {
    d = ismSlowOnce(X, Y, xInds1, xInds0, yInds1, yInds0)
    val = val + d
    # if (d != 0) { print("d"); print(d) }
  }
  return(val)
}

test_that("Check that hoeffding's D agrees with naive c++ code in 2 dims", {
  xInds0 = 0
  xInds1 = 0
  yInds0 = 0
  yInds1 = 0

  set.seed(123)

  for (i in 1:5) {
    X = matrix(rnorm(10), ncol = 1)
    Y = matrix(rnorm(10), ncol = 1)
    a = ismNaive(X, Y, xInds0, xInds1, yInds0, yInds1)
    b = hoeffD(X, Y)
    expect_equal(a, b)
  }

  for (i in 1:5) {
    X = matrix(rpois(10, lambda = 4), ncol = 1)
    Y = matrix(rpois(10, lambda = 4), ncol = 1)
    a = ismNaive(X, Y, xInds0, xInds1, yInds0, yInds1)
    b = hoeffD(X, Y)
    expect_equal(a, b)
  }
})

test_that("Check that hoeffding's D agrees with c++ code in 2 dims", {
  X = matrix(1:5, ncol = 1)
  n = 5
  flag = T
  for (i1 in 1:n) {
    for (i2 in 1:n) {
      for (i3 in 1:n) {
        for (i4 in 1:n) {
          for (i5 in 1:n) {
            Y = matrix(c(i1, i2, i3, i4, i5), ncol = 1)
            a = ismNaive(X, Y, 0, 0, 0, 0)
            b = ism(X, Y, 0, 0, 0, 0)
            flag = flag && (a == b)

            Y = matrix(c(i1, i2, i3, i4, i5), ncol = 1)
            a = ismNaive(Y, X, 0, 0, 0, 0)
            b = ism(Y, X, 0, 0, 0, 0)
            flag = flag && (a == b)
          }
        }
      }
    }
  }
  expect_true(flag)

  set.seed(123)
  for (i in 1:100) {
    X = matrix(rnorm(10), ncol = 1)
    Y = matrix(rnorm(10), ncol = 1)
    a = ismNaive(X, Y, 0, 0, 0, 0)
    b = ism(X, Y, 0, 0, 0, 0)
    expect_equal(a, b)

    X = matrix(rpois(10, lambda = 4), ncol = 1)
    Y = matrix(rpois(10, lambda = 4), ncol = 1)
    a = ismNaive(X, Y, 0, 0, 0, 0)
    b = ism(X, Y, 0, 0, 0, 0)
    expect_equal(a, b)
  }
})

test_that("Check that naive and fast methods agree in higher dims", {
  goodGaussSeeds = c(5279, 5283, 7287, 9633, 11819, 12584, 21095, 21123, 25437, 25950,
                27629, 28884, 28975, 29444, 30335, 30419, 30984, 31122, 33208,
                33981, 34292, 36879, 37560, 42528, 43121, 43693, 47386, 49991,
                51749, 56835)
  for (seed in goodGaussSeeds[1:15]) {
    set.seed(seed)
    xDim = 2
    yDim = 2
    n = 7
    X = matrix(rnorm(xDim * n), nrow = n)
    Y = matrix(rnorm(yDim * n), nrow = n)
    a = ismNaive(X, Y, 0, 1, 0, 1)
    b = ismSlow(X, Y, 0, 1, 0, 1)
    c = ism(X, Y, 0, 1, 0, 1)
    expect_true(a != 0)
    expect_equal(a, b)
    expect_equal(a, c)
  }

  goodPoisSeeds = c(1805, 2984, 3319, 3976, 8749, 10987, 11266, 11421, 11668,
                    11834, 11852, 12097, 12469, 15377, 15816, 17980, 18429,
                    19461, 20845, 22012, 22337, 22540, 23001, 23371, 24163,
                    24341, 24504, 24929, 25113, 25605)
  for (seed in goodPoisSeeds[1:15]) {
    set.seed(seed)
    xDim = 2
    yDim = 2
    n = 7
    X = matrix(rpois(xDim * n, lambda = 1), nrow = n)
    Y = matrix(rpois(yDim * n, lambda = 1), nrow = n)
    a = ismNaive(X, Y, 0, 1, 0, 1)
    b = ismSlow(X, Y, 0, 1, 0, 1)
    c = ism(X, Y, 0, 1, 0, 1)
    expect_true(a != 0)
    expect_equal(a, b)
    expect_equal(a, c)
  }

  goodSeedsHighDim1 = c(1272, 8513, 8712, 10147, 10272, 10535, 11722, 12754,
                        15628, 17812, 18232, 21691, 21796, 23354, 24403, 24458,
                        26167, 26539, 27899, 28420, 29552, 29941, 31652, 32960,
                        33785, 33922, 34205, 38176, 39013, 39729)
  for (i in goodSeedsHighDim1[1:15]) {
    set.seed(i)
    xDim = 3
    yDim = 3
    n = 20
    X = matrix(rpois(xDim * n, lambda = 2), nrow = n)
    Y = matrix(rpois(yDim * n, lambda = 2), nrow = n)
    a = ismNaive(X, Y, c(0,1), c(1,2), c(0,1), 2)
    b = ism(X, Y, c(0,1), c(1,2), c(0,1), 2)
    expect_true(a != 0)
    expect_equal(a, b)
  }

  goodSeedsHighDim2 = c(3306, 3846, 4919, 5608, 5880, 6940, 10384, 12361, 12588,
                        12618, 12664, 12958, 14526, 14755, 15241, 15352, 15357,
                        15384, 17610, 18598, 18666, 19889, 21334, 22328, 22995,
                        23699, 23997, 24426, 24888, 25119)
  for (i in goodSeedsHighDim2[1:15]) {
    set.seed(i)
    xDim = 3
    yDim = 3
    n = 20
    X = matrix(rpois(xDim * n, lambda = 2), nrow = n)
    Y = matrix(rpois(yDim * n, lambda = 2), nrow = n)
    a = ismNaive(X, Y, c(0,1), c(1,2), c(0,1), 1)
    b = ism(X, Y, c(0,1), c(1,2), c(0,1), 1)
    expect_true(a != 0)
    expect_equal(a, b)
  }
})

# a = sort(sample(1000000, 10000, replace = F))
# yar = c()
# for (i in a) {
#   set.seed(i)
#   xDim = 3
#   yDim = 3
#   n = 20
#   X = matrix(rpois(xDim * n, lambda = 2), nrow = n)
#   Y = matrix(rpois(yDim * n, lambda = 2), nrow = n)
#   if (ism(X, Y, c(0,1), c(1,2), c(0,1), 1) != 0) {
#     print(ism(X, Y, c(0,1), c(1,2), c(0,1), 1))
#     yar = c(yar, i)
#   }
#   if (length(yar) == 30) {
#     break
#   }
# }