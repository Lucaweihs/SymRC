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

#ifndef SymRC_NaiveUStatistics
#define SymRC_NaiveUStatistics

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

double approxNaiveUStatTime(const arma::mat& X, const arma::mat& Y,
                            const KernelEvaluator& kernel, int seconds);

/*********************************
 * SymRCKernelEvaluator
 *********************************/
class SymRCKernelEvaluator : public KernelEvaluator {
protected:
  const int ord;
  const arma::umat perms;
  const int xDim, yDim;
  const arma::umat posPerms;
  const arma::umat negPerms;

private:
  virtual bool minorIndicatorX(const arma::mat& vecs) const = 0;

  virtual bool minorIndicatorY(const arma::mat& vecs) const = 0;

public:
  SymRCKernelEvaluator(int xDim, int yDim,
                       const arma::umat& posPerms,
                       const arma::umat& negPerms);
  double eval(const arma::mat& X, const arma::mat& Y) const;
  int order() const;
};

class PartialTauStarKernelEvaluator : public SymRCKernelEvaluator {
private:
  bool minorIndicatorX(const arma::mat& vecs) const;
  bool minorIndicatorY(const arma::mat& vecs) const;

public:
  PartialTauStarKernelEvaluator(int xDim, int yDim);
};

class LexTauStarKernelEvaluator : public SymRCKernelEvaluator {
private:
  arma::uvec xPerm;
  arma::uvec yPerm;

  bool minorIndicator(const arma::mat& vecs, const arma::uvec& perm) const;
  bool minorIndicatorX(const arma::mat& vecs) const;
  bool minorIndicatorY(const arma::mat& vecs) const;

public:
  LexTauStarKernelEvaluator(int xDim, int yDim,
                            const arma::uvec& xPerm,
                            const arma::uvec& yPerm);
};

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
