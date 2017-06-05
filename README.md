# SymRC Package

## Purpose

This package allows for the efficient estimation of symmetric rank covariances (SRCs) which, themselves, are measures of dependence between collections of random variables. SRCs include, for example, Hoeffding's D, Hoeffding's R, and several multivariate extensions of the Bergsma-Dassios sign-covariance tau*. For more information, see the help for the functions

* `hoeffD`
* `hoeffR`
* `pTStar`
* `jTStar`

## Examples

Lets run a simple permutation test for dependence between two 1-dimensional random variables X and Y.
```
# The data
n = 100
X = rnorm(n)
Y = X^2 + rnorm(n)

# Permutation test using the various measures
set.seed(123)
reps = 500
jTStarValues = numeric(reps)
hoeffDValues = numeric(reps)
for (i in 1:reps) {
  permutedY = sample(Y)
  jTStarValues[i] = jTStar(X, permutedY)
  hoeffDValues[i] = hoeffD(X, permutedY)
}
pValueForJTStar = mean(jTStarValues >= jTStar(X, Y)) # = 0
pValueForHoeffD = mean(hoeffDValues >= hoeffD(X, Y)) # = 0
```

Now lets run another permutation test where X is 2-dimensional and Y is 1-dimensional (Y will equal the XOR of the X entries).
```
# The data
n = 30
X = matrix(rbinom(2 * n, size = 1, p = 1/2), ncol = 2)
Y = apply(X, MARGIN = 1, FUN = function(x) { 
  x[1] + x[2] - 2 * x[1] * x[2] 
})

# Permutation test using pTStar and hoeffR, this takes some time.
set.seed(123)
reps = 250
pTStarValues = numeric(reps)
hoeffRValues = numeric(reps)
for (i in 1:reps) {
  print(i)
  permutedY = sample(Y)
  pTStarValues[i] = pTStar(X, permutedY, method = "range-tree")
  hoeffRValues[i] = hoeffR(X, permutedY)
}
pValueForPRStar = mean(pTStarValues >= pTStar(X, Y)) # = 0.008
pValueForHoeffR = mean(hoeffRValues >= hoeffR(X, Y)) # = 0.012
```