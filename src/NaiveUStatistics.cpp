/***
 * Copyright (C) 2016 Luca Weihs
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "RcppArmadilloExtensions/sample.h"
#include "NaiveUStatistics.h"
#include "HelperFunctions.h"
#include <iostream>

double naiveUStatRecurse(const arma::mat& X, const arma::mat& Y,
                         const KernelEvaluator& kernel,
                         int dim, int index,
                         arma::uvec& currentIndex) {
  int n = X.n_rows;
  int order = kernel.order();

  double val = 0.0;
  if (dim == order - 1) {
    while (index < n) {
      currentIndex(dim) = index;
      val += kernel.eval(X.rows(currentIndex), Y.rows(currentIndex));
      index++;
    }
  // } else if (order - dim - 1 == 5) {
  //   currentIndex(dim) = index;
  //   for (int i1 = index + 1; i1 < order - 3; i1++) {
  //     currentIndex(dim + 1) = i1;
  //     for (int i2 = i1 + 1; i2 < order - 2; i2++) {
  //       currentIndex(dim + 2) = i2;
  //       for (int i3 = i2 + 1; i3 < order - 1; i3++) {
  //         currentIndex(dim + 3) = i3;
  //         for (int i4 = i3 + 1; i4 < order; i4++) {
  //           currentIndex(dim + 4) = i4;
  //           val += kernel.eval(X.rows(currentIndex), Y.rows(currentIndex));
  //         }
  //       }
  //     }
  //   }
  //   return val;
  } else {
    while (index + (order - (dim + 1)) < n) {
      // While there enough datapoints >index to fill up
      // the remaining dimensions, add up the recursed values.
      currentIndex(dim) = index;
      val += naiveUStatRecurse(X, Y, kernel, dim + 1, index + 1,
                               currentIndex);
      index++;
    }
  }
  return val;
}

double naiveUStat(const arma::mat& X, const arma::mat& Y,
                  const KernelEvaluator& kernel) {
  int order = kernel.order();
  int n = X.n_rows;
  if (order > n) {
    throw Rcpp::exception("Number of samples must be > kernel order.");
  }

  arma::uvec index = arma::zeros<arma::uvec>(order);
  return naiveUStatRecurse(X, Y, kernel, 0, 0, index) / nChooseM(1.0 * n, 1.0 * order);
}

double approxNaiveUStat(const arma::mat& X, const arma::mat& Y,
                        const KernelEvaluator& kernel, int sims) {
  int order = kernel.order();
  int n = X.n_rows;

  if (order > n) {
    throw Rcpp::exception("Number of samples must be > kernel order.");
  }

  double val = 0.0;
  arma::uvec allInds = arma::zeros<arma::uvec>(n);
  for (int i = 0; i < n; i++) {
    allInds(i) = i;
  }
  for (int i = 0; i < sims; i++) {
    arma::uvec inds = Rcpp::RcppArmadillo::sample(allInds, order, false);
    val += kernel.eval(X.rows(inds), Y.rows(inds));
  }

  return val / sims;
}

/******************************
 * Example U-statistic kernels.
 *******************************/
KendallsTauEvaluator::KendallsTauEvaluator() {}

int KendallsTauEvaluator::order() const {
  return ord;
}

double KendallsTauEvaluator::eval(const arma::mat& X, const arma::mat& Y) const {
  if (((X(0,0) < X(1,0)) && (Y(0,0) < Y(1,0))) ||
      ((X(1,0) < X(0,0)) && (Y(1,0) < Y(0,0)))) {
    return 1.0;
  } else if (((X(0,0) < X(1,0)) && (Y(1,0) < Y(0,0))) ||
    ((X(1,0) < X(0,0)) && (Y(0,0) < Y(1,0)))) {
    return -1.0;
  } else {
    return 0.0;
  }
}

SpearmansRhoEvaluator::SpearmansRhoEvaluator() {}

int SpearmansRhoEvaluator::order() const {
  return ord;
}

double SpearmansRhoEvaluator::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::uvec sortIndex = arma::sort_index(X.col(0));
  arma::vec sortedX = X.col(0);
  sortedX = sortedX.elem(sortIndex);
  if (sortedX(0) == sortedX(1) || sortedX(1) == sortedX(2)) {
    Rcpp::stop("This example spearman's rho only works with untied data.");
  }

  arma::vec newY = Y.col(0);
  newY = newY.elem(sortIndex);

  if (std::max(newY(0), newY(1)) < newY(2) ||
      newY(0) < std::min(newY(1), newY(2))) {
    return 1.0;
  } else if (std::min(newY(0), newY(1)) > newY(2) ||
    newY(0) > std::max(newY(1), newY(2))) {
    return -1.0;
  }
  return 0.0;
}
