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

#ifndef WCM_MultivariateTauStar
#define WCM_MultivariateTauStar

#include "NaiveUStatistics.h"
#include "EmpiricalDistribution.h"

class GenericTauStarKernelEvaluator : public KernelEvaluator {
protected:
  static const int ord = 4;
  static const arma::umat perms;
  int xDim, yDim;

private:
  virtual bool minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                      const arma::vec& v2, const arma::vec& v3) const = 0;

  virtual bool minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                               const arma::vec& v2, const arma::vec& v3) const = 0;

public:
  GenericTauStarKernelEvaluator(int xDim, int yDim);
  int order() const;
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

class PartialTauStarKernelEvaluator : public GenericTauStarKernelEvaluator {
private:
  bool minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                      const arma::vec& v2, const arma::vec& v3) const;
  bool minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const;
public:
  PartialTauStarKernelEvaluator(int xDim, int yDim);
};

class LexTauStarKernelEvaluator : public GenericTauStarKernelEvaluator {
private:
  arma::uvec xPerm;
  arma::uvec yPerm;

  bool minorIndicator(const arma::vec& v0, const arma::vec& v1,
                      const arma::vec& v2, const arma::vec& v3,
                      const arma::uvec& perm) const;
  bool minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const;
  bool minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const;

public:
  LexTauStarKernelEvaluator(int xDim, int yDim, const arma::uvec& xPerm,
                            const arma::uvec& yPerm);
};


class FullLexTauStarKernelEvaluator : public KernelEvaluator {
private:
  static const int ord = 4;
  std::vector<LexTauStarKernelEvaluator> evaluators;
  int xDim, yDim;

public:
  FullLexTauStarKernelEvaluator(int xDim, int yDim);
  double eval(const arma::mat& X, const arma::mat& Y) const;
  int order() const;
};

class PartialTauStarEvaluator {
private:
  int xDim, yDim;
  double countGreaterEqInX(const arma::vec& x,
                         const EmpiricalDistribution& ed) const;
  double countGreaterEqInY(const arma::vec& y,
                         const EmpiricalDistribution& ed) const;
  double countGreaterEqInXY(const arma::vec& x, const arma::vec& y,
                          const EmpiricalDistribution& ed) const;
  double countLesserEqInY(const arma::vec& y,
                          const EmpiricalDistribution& ed) const;
  double countGreaterEqXLesserEqY(const arma::vec& x, const arma::vec& y,
                                  const EmpiricalDistribution& ed) const;

  double posConCount(const arma::vec& x0, const arma::vec& x1,
                     const arma::vec& y0, const arma::vec& y1,
                     const EmpiricalDistribution& ed) const;

  double negConCount(const arma::vec& x0, const arma::vec& x1,
                     const arma::vec& y0, const arma::vec& y1,
                     const EmpiricalDistribution& ed) const;

  double disCount(const arma::vec& x0, const arma::vec& x1,
                  const arma::vec& y0, const arma::vec& y1,
                  const EmpiricalDistribution& ed) const;

  EmpiricalDistribution createPairsED(const arma::mat& X,
                                            const arma::mat& Y) const;

  EmpiricalDistribution createIncomparableED(const arma::mat& X,
                                             const arma::mat& Y) const;

public:
  PartialTauStarEvaluator(int xDim, int yDim);
  double eval(const arma::mat& X, const arma::mat& Y) const;
};


class JointTauStarKernelEvaluator : public GenericTauStarKernelEvaluator {
private:
  arma::uvec xOnOffVec;
  arma::uvec yOnOffVec;
  bool minorIndicator(const arma::vec& v0, const arma::vec& v1,
                             const arma::vec& v2, const arma::vec& v3,
                             const arma::uvec& onOffVec) const;
  bool minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const;
  bool minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const;

public:
  JointTauStarKernelEvaluator(const arma::uvec& xOnOffVec,
                              const arma::uvec& yOnOffVec);
};

class JointTauStarEvaluator {
private:
  static bool lessInPartialOrder(const arma::vec& v0,
                                 const arma::vec& v1,
                                 const arma::uvec& onOffVec);
  int xDim, yDim;
  arma::uvec xOnOffVec;
  arma::uvec yOnOffVec;

  EmpiricalDistribution createComparableED(const arma::mat& X,
                                           const arma::mat& Y) const;

  double posConCount(const arma::vec& x0, const arma::vec& x1,
                     const arma::vec& y0, const arma::vec& y1,
                     const EmpiricalDistribution& ed) const;

  double negConCount(const arma::vec& x0, const arma::vec& x1,
                     const arma::vec& y0, const arma::vec& y1,
                     const EmpiricalDistribution& ed) const;

  double disCount(const arma::vec& x0, const arma::vec& x1,
                  const arma::vec& y0, const arma::vec& y1,
                  const EmpiricalDistribution& ed) const;

public:
  JointTauStarEvaluator(const arma::uvec& xOnOffVec,
                        const arma::uvec& yOnOffVec);
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

#endif
