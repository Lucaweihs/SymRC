hoeffDSuperSlow <- function(x, y) {
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

hoeffRSuperSlow <- function(x, y) {
  n = length(x)
  val = 0;
  for (i1 in 1:n) {
    for (i2 in 1:n) {
      for (i3 in 1:n) {
        for (i4 in 1:n) {
          for (i5 in 1:n) {
            for (i6 in 1:n) {
              if (length(unique(c(i1,i2,i3,i4,i5,i6))) == 6) {
                if (max(x[i1], x[i2]) <= x[i5] &&
                    x[i5] < min(x[i3], x[i4])) {
                  val = val + (max(y[i1], y[i2]) <= y[i6] && y[i6] < min(y[i3], y[i4]))
                  val = val + (max(y[i3], y[i4]) <= y[i6] && y[i6] < min(y[i1], y[i2]))
                  val = val - 2 * (max(y[i1], y[i3]) <= y[i6] && y[i6] < min(y[i2], y[i4]))
                }
              }
            }
          }
        }
      }
    }
  }
  return(val / (choose(n, 6) * factorial(6)))
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

expect_all_equal <- function(...) {
  things = list(...)
  if (length(things) <= 1) {
    stop("expect_all_equal requires at least 2 arguments.")
  }
  thing1 = things[[1]]
  for (i in 2:length(things)) {
    expect_equal(thing1, things[[i]])
  }
}