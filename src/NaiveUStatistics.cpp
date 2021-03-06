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
#include <chrono>

typedef SymRCKernelEvaluator SymRCKE;

double naiveUStatRecurse(const arma::mat& X, const arma::mat& Y,
                         const KernelEvaluator& kernel,
                         int dim, int index,
                         arma::uvec& currentIndex,
                         unsigned int& sims) {
  int n = X.n_rows;
  int order = kernel.order();

  double val = 0.0;
  if (dim == order - 1) {
    while (index < n) {
      if (sims % 600000 == 0) {
        Rcpp::checkUserInterrupt();
      }
      currentIndex(dim) = index;
      val += kernel.eval(X.rows(currentIndex), Y.rows(currentIndex));
      index++;
      sims++;
    }
  } else {
    while (index + (order - (dim + 1)) < n) {
      // While there enough datapoints >index to fill up
      // the remaining dimensions, add up the recursed values.
      currentIndex(dim) = index;
      val += naiveUStatRecurse(X, Y, kernel, dim + 1, index + 1,
                               currentIndex, sims);
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
    throw Rcpp::exception("Number of samples must be >= kernel order.");
  }

  arma::uvec index = arma::zeros<arma::uvec>(order);
  unsigned int sims = 0;
  return naiveUStatRecurse(X, Y, kernel, 0, 0, index, sims) / nChooseM(1.0 * n, 1.0 * order);
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
    if (sims % 600000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    arma::uvec inds = Rcpp::RcppArmadillo::sample(allInds, order, false);
    val += kernel.eval(X.rows(inds), Y.rows(inds));
  }

  return val / sims;
}

double approxNaiveUStatTime(const arma::mat& X, const arma::mat& Y,
                        const KernelEvaluator& kernel, int seconds) {
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

  int sims = 0;
  auto start = std::chrono::system_clock::now();
  while (true) {
    if (sims % 100000 == 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - start).count() > 1000 * seconds) {
      Rcpp::Rcout << std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now() - start).count() << std::endl;
      break;
    }
    if (sims % 600000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    arma::uvec inds = Rcpp::RcppArmadillo::sample(allInds, order, false);
    val += kernel.eval(X.rows(inds), Y.rows(inds));
    sims++;
  }
  Rcpp::Rcout << sims << " simulations completed in given time." << std::endl;
  return val / sims;
}

/*********************************
 * SymRCKernelEvaluator
 *********************************/

SymRCKE::SymRCKernelEvaluator(int xDim, int yDim,
                              const arma::umat& posPerms,
                              const arma::umat& negPerms):
  ord(posPerms.n_cols), perms(permutations(ord)),
  xDim(xDim), yDim(yDim), posPerms(posPerms), negPerms(negPerms) {
  if (posPerms.n_rows != negPerms.n_rows) {
    throw Rcpp::exception("negPerms and posPerms must have "
                            "the same number of columns");
  }
}

int SymRCKE::order() const {
  return ord;
}

double SymRCKE::eval(const arma::mat& X, const arma::mat& Y) const {
  double fullSum = 0;
  for (int i = 0; i < perms.n_rows; i++) {
    if (!minorIndicatorX(X.rows(perms.row(i)))) {
      continue;
    }
    arma::mat subY = Y.rows(perms.row(i));

    for (int i = 0; i < posPerms.n_rows; i++) {
      fullSum += minorIndicatorY(subY.rows(posPerms.row(i))) ? 1.0 : 0.0;
    }
    for (int i = 0; i < negPerms.n_rows; i++) {
      fullSum -= minorIndicatorY(subY.rows(negPerms.row(i))) ? 1.0 : 0.0;
    }
  }
  return 2 * posPerms.n_rows * fullSum / perms.n_rows;
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
