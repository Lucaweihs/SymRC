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

#ifndef WCM_NaiveUStatistics
#define WCM_NaiveUStatistics

// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

class KernelEvaluator {
public:
  virtual int order() const = 0;
  virtual double eval(const arma::mat& X, const arma::mat& Y) const = 0;
};

double naiveUStat(const arma::mat& X, const arma::mat& Y,
                  const KernelEvaluator& kernel);

double approxNaiveUStat(const arma::mat& X, const arma::mat& Y,
                        const KernelEvaluator& kernel, int sims);

/******************************
 * Example U-statistic kernels.
 *******************************/
/**
 * Kernel Evaluator for Kendall's Tau
 */
class KendallsTauEvaluator : public KernelEvaluator {
private:
  static const int ord = 2;

public:
  KendallsTauEvaluator();
  int order() const;
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

/**
 * Kernel Evaluator for Spearmans's Rho
 */
class SpearmansRhoEvaluator : public KernelEvaluator {
private:
  static const int ord = 3;

public:
  SpearmansRhoEvaluator();
  int order() const;
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

#endif
