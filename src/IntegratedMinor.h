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

#ifndef WCM_IntegratedMinor
#define WCM_IntegratedMinor

#include "NaiveUStatistics.h"
#include "EmpiricalDistribution.h"

class SubsetMinorPartition {
private:
  arma::uvec inds0;
  arma::uvec inds1;
  arma::uvec allInds;
  arma::uvec intersection;
  arma::uvec complement;
  arma::uvec counts;
  arma::uvec symmetricDiff;
  arma::uvec inds0Unique;
  arma::uvec inds1Unique;

public:
  SubsetMinorPartition(int dim, arma::uvec leftInds, arma::uvec rightInds);

  const arma::uvec& getAllInds() const;
  const arma::uvec& getNonUniqueLeftInds() const;
  const arma::uvec& getNonUniqueRightInds() const;
  const arma::uvec& getUniqueLeftInds() const;
  const arma::uvec& getUniqueRightInds() const;
  const arma::uvec& getIntersection() const;
  const arma::uvec& getCounts() const;
  const arma::uvec& getComplement() const;
  const arma::uvec& getSymmetricDiff() const;
};

class IntegratedMinorKernelEvaluator : public KernelEvaluator {
private:
  static const int ord = 5;
  SubsetMinorPartition xPart;
  SubsetMinorPartition yPart;
  arma::umat perms;

  bool minorIndicator(const arma::vec& v0, const arma::vec& v1,
                      const arma::vec& v2, const arma::vec& v3,
                      const SubsetMinorPartition& part) const;

  bool weightIndicator(const arma::mat& V, const SubsetMinorPartition& part) const;

public:
  IntegratedMinorKernelEvaluator(int xDim, int yDim,
                           arma::uvec xInds0, arma::uvec xInds1,
                           arma::uvec yInds0, arma::uvec yInds1);
  int order() const;
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

class IntegratedMinorEvaluator {
private:
  SubsetMinorPartition xPart;
  SubsetMinorPartition yPart;

  std::vector<double> minLower;

  int xDim;
  int yDim;

  bool xIndsEq;
  bool yIndsEq;

  double countGreaterInX(const std::vector<double> point,
                         const arma::uvec& xGreaterInds,
                         const EmpiricalDistribution& ed) const;

  double countGreaterInY(const std::vector<double> point,
                         const arma::uvec& yGreaterInds,
                         const EmpiricalDistribution& ed) const;

  double countGreaterInXY(const std::vector<double> point,
                          const arma::uvec& xGreaterInds,
                          const arma::uvec& yGreaterInds,
                          const EmpiricalDistribution& ed) const;

public:
  IntegratedMinorEvaluator(int xDim, int yDim,
                           arma::uvec xInds0, arma::uvec xInds1,
                           arma::uvec yInds0, arma::uvec yInds1);

  double eval(const arma::mat& X, const arma::mat& Y) const;
};

#endif

