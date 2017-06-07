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

#include "NaiveUStatistics.h"
#include "IntegratedMinor.h"
#include "MultivariateTauStar.h"
#include "RcppArmadillo.h"
#include "HelperFunctions.h"
#include "OrthogonalRangeQuerier.h"
#include "Hoeffding.h"

typedef std::shared_ptr<OrthogonalRangeQuerierBuilder> shr_ptrORQB;

// [[Rcpp::export]]
double hoeffdingROrthTensor(const arma::mat& X, const arma::mat& Y) {
  HoeffdingREvaluator hre(X.n_cols, Y.n_cols,
                          shr_ptrORQB(new OrthogonalRangeTensorBuilder()));
  return hre.eval(X, Y);
}

// [[Rcpp::export]]
double hoeffdingRNaive(const arma::mat& X, const arma::mat& Y) {
  HoeffdingREvaluator hre(X.n_cols, Y.n_cols,
                          shr_ptrORQB(new NaiveRangeCounterBuilder()));
  return hre.eval(X, Y);
}

// [[Rcpp::export]]
double hoeffdingRFromDef(const arma::mat& X, const arma::mat& Y) {
  HoeffdingRKernelEvaluator hrke(X.n_cols, Y.n_cols);
  return 0.25 * naiveUStat(X, Y, hrke);
}

// [[Rcpp::export]]
double hoeffdingDRangeTree(const arma::mat& X, const arma::mat& Y) {
  HoeffdingDEvaluator hde(X.n_cols, Y.n_cols,
                          shr_ptrORQB(
                            new AlignedRangeTreeBuilder()));
  return hde.eval(X, Y);
}

// [[Rcpp::export]]
double hoeffdingDNaive(const arma::mat& X, const arma::mat& Y) {
  HoeffdingDEvaluator hde(X.n_cols, Y.n_cols,
                          shr_ptrORQB(
                            new NaiveRangeCounterBuilder()));
  return hde.eval(X, Y);
}

// [[Rcpp::export]]
double hoeffdingDFromDef(const arma::mat& X, const arma::mat& Y) {
  HoeffdingDKernelEvaluator hdke(X.n_cols, Y.n_cols);
  return 0.25 * naiveUStat(X, Y, hdke);
}

// [[Rcpp::export]]
double jointTauStarRangeTree(const arma::mat& X, const arma::mat& Y,
                         const arma::uvec& xOnOffVec,
                         const arma::uvec& yOnOffVec) {
  JointTauStarEvaluator jtse(
      xOnOffVec, yOnOffVec,
      shr_ptrORQB(new AlignedRangeTreeBuilder()));
  return jtse.eval(X, Y);
}

// [[Rcpp::export]]
double jointTauStarNaive(const arma::mat& X, const arma::mat& Y,
                    const arma::uvec& xOnOffVec,
                    const arma::uvec& yOnOffVec) {
  JointTauStarEvaluator jtse(
      xOnOffVec, yOnOffVec,
      shr_ptrORQB(new NaiveRangeCounterBuilder()));
  return jtse.eval(X, Y);
}

// [[Rcpp::export]]
double jointTauStarFromDef(const arma::mat& X, const arma::mat& Y,
                         const arma::uvec& xOnOffVec,
                         const arma::uvec& yOnOffVec) {
  JointTauStarKernelEvaluator jtske(xOnOffVec, yOnOffVec);
  return naiveUStat(X, Y, jtske);
}

// [[Rcpp::export]]
double jointTauStarApprox(const arma::mat& X, const arma::mat& Y,
                               const arma::uvec& xOnOffVec,
                               const arma::uvec& yOnOffVec,
                               int sims) {
  JointTauStarKernelEvaluator jtske(xOnOffVec, yOnOffVec);
  return approxNaiveUStat(X, Y, jtske, sims);
}

// [[Rcpp::export]]
double fullLexTauStarFromDef(const arma::mat& X, const arma::mat& Y) {
  FullLexTauStarKernelEvaluator fltske(X.n_cols, Y.n_cols);
  return naiveUStat(X, Y, fltske);
}

// [[Rcpp::export]]
double fullLexTauStarApprox(const arma::mat& X, const arma::mat& Y,
                             int sims) {
  FullLexTauStarKernelEvaluator fltske(X.n_cols, Y.n_cols);
  return approxNaiveUStat(X, Y, fltske, sims);
}

// [[Rcpp::export]]
double lexTauStarFromDef(const arma::mat& X, const arma::mat& Y,
                       const arma::uvec& xPerm, const arma::uvec& yPerm) {
  LexTauStarKernelEvaluator ltske(X.n_cols, Y.n_cols, xPerm, yPerm);
  return naiveUStat(X, Y, ltske);
}

// [[Rcpp::export]]
double lexTauStarApprox(const arma::mat& X, const arma::mat& Y,
                             const arma::uvec& xPerm, const arma::uvec& yPerm,
                             int sims) {
  LexTauStarKernelEvaluator ltske(X.n_cols, Y.n_cols, xPerm, yPerm);
  return approxNaiveUStat(X, Y, ltske, sims);
}

// [[Rcpp::export]]
double partialTauStarRangeTree(const arma::mat& X, const arma::mat& Y) {
  PartialTauStarEvaluator ptse(
      X.n_cols, Y.n_cols,
      shr_ptrORQB(new AlignedRangeTreeBuilder()));
  return ptse.eval(X, Y);
}

// [[Rcpp::export]]
double partialTauStarNaive(const arma::mat& X, const arma::mat& Y) {
  PartialTauStarEvaluator ptse(
      X.n_cols, Y.n_cols,
      shr_ptrORQB(new NaiveRangeCounterBuilder()));
  return ptse.eval(X, Y);
}

// [[Rcpp::export]]
double partialTauStarFromDef(const arma::mat& X, const arma::mat& Y) {
  PartialTauStarKernelEvaluator ptske(X.n_cols, Y.n_cols);
  return naiveUStat(X, Y, ptske);
}

// [[Rcpp::export]]
double partialTauStarApprox(const arma::mat& X, const arma::mat& Y,
                                 int sims) {
  PartialTauStarKernelEvaluator ptske(X.n_cols, Y.n_cols);
  return approxNaiveUStat(X, Y, ptske, sims);
}

// [[Rcpp::export]]
double ismRangeTree(const arma::mat& X, const arma::mat& Y,
           const arma::uvec& xInds0, const arma::uvec& xInds1,
           const arma::uvec& yInds0, const arma::uvec& yInds1) {
  IntegratedMinorEvaluator ime(X.n_cols, Y.n_cols, xInds0, xInds1,
                               yInds0, yInds1);
  return ime.eval(X, Y);
}

// [[Rcpp::export]]
double ismFromDef(const arma::mat& X, const arma::mat& Y,
                const arma::uvec& xInds0, const arma::uvec& xInds1,
                const arma::uvec& yInds0, const arma::uvec& yInds1) {
  IntegratedMinorKernelEvaluator imke(X.n_cols, Y.n_cols, xInds0, xInds1,
                               yInds0, yInds1);
  return naiveUStat(X, Y, imke);
}

// [[Rcpp::export]]
double ismApprox(const arma::mat& X, const arma::mat& Y,
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
double kendallsTauApprox(const arma::mat& X,
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
double spearmansRhoApprox(const arma::mat& X,
                               const arma::mat& Y,
                               int sims) {
  SpearmansRhoEvaluator sr;
  return approxNaiveUStat(X, Y, sr, sims);
}

// [[Rcpp::export]]
arma::uvec orthRangeTensorCount(const arma::mat& samples,
                         const arma::umat& lowerMat,
                         const arma::umat& upperMat) {
  arma::umat jointRanks = toJointRankMatrix(samples);
  arma::uvec counts(lowerMat.n_rows);
  OrthogonalRangeTensor ort(jointRanks);
  for (int i = 0; i < lowerMat.n_rows; i++) {
    counts(i) = ort.countInRange(lowerMat.row(i).t(), upperMat.row(i).t());
  }
  return counts;
}

// [[Rcpp::export]]
arma::uvec alignedRangeTreeCount(const arma::mat& samples,
                                 const arma::umat& lowerMat,
                                 const arma::umat& upperMat) {
  arma::umat jointRanks = toJointRankMatrix(samples);
  arma::uvec counts(lowerMat.n_rows);
  AlignedRangeTree art(jointRanks);
  for (int i = 0; i < lowerMat.n_rows; i++) {
    counts(i) = art.countInRange(lowerMat.row(i).t(), upperMat.row(i).t());
  }
  return counts;
}


