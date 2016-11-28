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

// [[Rcpp::plugins(cpp11)]]
#include "NaiveUStatistics.h"
#include "IntegratedMinor.h"
#include "MultivariateTauStar.h"
#include "RcppArmadillo.h"

// [[Rcpp::export]]
double partialTauStar(const arma::mat& X, const arma::mat& Y) {
  PartialTauStarEvaluator ptse(X.n_cols, Y.n_cols);
  return ptse.eval(X, Y);
}

// [[Rcpp::export]]
double fullLexTauStarNaive(const arma::mat& X, const arma::mat& Y) {
  FullLexTauStarKernelEvaluator fltske(X.n_cols, Y.n_cols);
  return naiveUStat(X, Y, fltske);
}

// [[Rcpp::export]]
double fullLexTauStarNaiveApprox(const arma::mat& X, const arma::mat& Y,
                             int sims) {
  FullLexTauStarKernelEvaluator fltske(X.n_cols, Y.n_cols);
  return approxNaiveUStat(X, Y, fltske, sims);
}

// [[Rcpp::export]]
double lexTauStarNaive(const arma::mat& X, const arma::mat& Y,
                       const arma::uvec& xPerm, const arma::uvec& yPerm) {
  LexTauStarKernelEvaluator ltske(X.n_cols, Y.n_cols, xPerm, yPerm);
  return naiveUStat(X, Y, ltske);
}

// [[Rcpp::export]]
double lexTauStarNaiveApprox(const arma::mat& X, const arma::mat& Y,
                             const arma::uvec& xPerm, const arma::uvec& yPerm,
                             int sims) {
  LexTauStarKernelEvaluator ltske(X.n_cols, Y.n_cols, xPerm, yPerm);
  return approxNaiveUStat(X, Y, ltske, sims);
}

// [[Rcpp::export]]
double partialTauStarNaive(const arma::mat& X, const arma::mat& Y) {
  PartialTauStarKernelEvaluator ptske(X.n_cols, Y.n_cols);
  return naiveUStat(X, Y, ptske);
}

// [[Rcpp::export]]
double partialTauStarNaiveApprox(const arma::mat& X, const arma::mat& Y,
                                 int sims) {
  PartialTauStarKernelEvaluator ptske(X.n_cols, Y.n_cols);
  return approxNaiveUStat(X, Y, ptske, sims);
}

// [[Rcpp::export]]
double ism(const arma::mat& X, const arma::mat& Y,
           const arma::uvec& xInds0, const arma::uvec& xInds1,
           const arma::uvec& yInds0, const arma::uvec& yInds1) {
  IntegratedMinorEvaluator ime(X.n_cols, Y.n_cols, xInds0, xInds1,
                               yInds0, yInds1);
  return ime.eval(X, Y);
}

// [[Rcpp::export]]
double ismNaive(const arma::mat& X, const arma::mat& Y,
                const arma::uvec& xInds0, const arma::uvec& xInds1,
                const arma::uvec& yInds0, const arma::uvec& yInds1) {
  IntegratedMinorKernelEvaluator imke(X.n_cols, Y.n_cols, xInds0, xInds1,
                               yInds0, yInds1);
  return naiveUStat(X, Y, imke);
}

// [[Rcpp::export]]
double ismNaiveApprox(const arma::mat& X, const arma::mat& Y,
                      const arma::uvec& xInds0, const arma::uvec& xInds1,
                      const arma::uvec& yInds0, const arma::uvec& yInds1,
                      int sims) {
  IntegratedMinorKernelEvaluator imke(X.n_cols, Y.n_cols, xInds0, xInds1,
                               yInds0, yInds1);
  return approxNaiveUStat(X, Y, imke, sims);
}

// [[Rcpp::export]]
double kendallsTauNaive(const arma::mat& X, const arma::mat& Y) {
  KendallsTauEvaluator ke;
  return naiveUStat(X, Y, ke);
}

// [[Rcpp::export]]
double kendallsTauNaiveApprox(const arma::mat& X,
                              const arma::mat& Y,
                              int sims) {
  KendallsTauEvaluator ke;
  return approxNaiveUStat(X, Y, ke, sims);
}

// [[Rcpp::export]]
double spearmansRhoNaive(const arma::mat& X, const arma::mat& Y) {
  SpearmansRhoEvaluator sr;
  return naiveUStat(X, Y, sr);
}

// [[Rcpp::export]]
double spearmansRhoNaiveApprox(const arma::mat& X,
                               const arma::mat& Y,
                               int sims) {
  SpearmansRhoEvaluator sr;
  return approxNaiveUStat(X, Y, sr, sims);
}