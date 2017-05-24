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

#include "MultivariateTauStar.h"
#include "HelperFunctions.h"
#include <chrono>

typedef GenericTauStarKernelEvaluator GTSKE;
typedef PartialTauStarKernelEvaluator PTSKE;
typedef LexTauStarKernelEvaluator LTSKE;
typedef FullLexTauStarKernelEvaluator FLTSKE;
typedef PartialTauStarEvaluator PTSE;
typedef JointTauStarKernelEvaluator JTSKE;
typedef JointTauStarEvaluator JTSE;

const arma::umat GTSKE::perms = permutations(4);

/*********************************
 * GenericTauStarKernelEvaluator
 *********************************/

GTSKE::GenericTauStarKernelEvaluator(int xDim, int yDim):
  xDim(xDim), yDim(yDim) {}

int GTSKE::order() const {
  return ord;
}

double GTSKE::eval(const arma::mat& X, const arma::mat& Y) const {
  double fullSum = 0;
  for (int i = 0; i < perms.n_rows; i++) {
    arma::vec x0 = X.row(perms(i,0)).t();
    arma::vec x1 = X.row(perms(i,1)).t();
    arma::vec x2 = X.row(perms(i,2)).t();
    arma::vec x3 = X.row(perms(i,3)).t();

    if (!minorIndicatorX(x0, x1, x2, x3)) {
      continue;
    }

    arma::vec y0 = Y.row(perms(i,0)).t();
    arma::vec y1 = Y.row(perms(i,1)).t();
    arma::vec y2 = Y.row(perms(i,2)).t();
    arma::vec y3 = Y.row(perms(i,3)).t();

    if (minorIndicatorY(y0, y1, y2, y3)) {
      fullSum += 1.0;
    }
    if (minorIndicatorY(y2, y3, y0, y1)) {
      fullSum += 1.0;
    }
    if (minorIndicatorY(y0, y2, y1, y3)) {
      fullSum += -2.0;
    }
  }
  return fullSum / perms.n_rows;
}

/*********************************
 * PartialTauStarKernelEvaluator
 *********************************/

PTSKE::PartialTauStarKernelEvaluator(int xDim, int yDim):
  GenericTauStarKernelEvaluator(xDim, yDim) {}

bool PTSKE::minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                      const arma::vec& v2, const arma::vec& v3) const {
  bool flag2 = false;
  bool flag3 = false;
  for (int i = 0; (!flag2 || !flag3) && (i < v0.size()); i++) {
    flag2 = flag2 || v0(i) < v2(i);
    flag3 = flag3 || v0(i) < v3(i);
  }
  if (!flag2 || !flag3) {
    return false;
  }

  flag2 = false;
  flag3 = false;
  for (int i = 0; (!flag2 || !flag3) && (i < v0.size()); i++) {
    flag2 = flag2 || v1(i) < v2(i);
    flag3 = flag3 || v1(i) < v3(i);
  }
  if (!flag2 || !flag3) {
    return false;
  }
  return true;
}

bool PTSKE::minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                            const arma::vec& v2, const arma::vec& v3) const {
  return minorIndicatorX(v0, v1, v2, v3);
}

/*********************************
 * LexTauStarKernelEvaluator
 *********************************/

LTSKE::LexTauStarKernelEvaluator(int xDim, int yDim,
                                 const arma::uvec& xPerm,
                                 const arma::uvec& yPerm) :
  GenericTauStarKernelEvaluator(xDim, yDim), xPerm(xPerm), yPerm(yPerm) {}

bool lexLessThan(const arma::vec& v0, const arma::vec& v1,
                 const arma::uvec& perm) {
  for (int i = 0; i < perm.size(); i++) {
    if (v0(perm(i)) > v1(perm(i))) {
      return false;
    } else if (v0(perm(i)) < v1(perm(i))) {
      return true;
    }
  }
  return false;
}

bool lexLessThan(const arma::vec& v0, const arma::vec& v1) {
  for (int i = 0; i < v0.size(); i++) {
    if (v0(i) > v1(i)) {
      return false;
    } else if (v0(i) < v1(i)) {
      return true;
    }
  }
  return false;
}

bool LTSKE::minorIndicator(const arma::vec& v0, const arma::vec& v1,
                            const arma::vec& v2, const arma::vec& v3,
                            const arma::uvec& perm) const {
  return lexLessThan(v0, v2, perm) && lexLessThan(v0, v3, perm) &&
    lexLessThan(v1, v2, perm) && lexLessThan(v1, v3, perm);
}

bool LTSKE::minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                           const arma::vec& v2, const arma::vec& v3) const {
  return minorIndicator(v0, v1, v2, v3, xPerm);
}

bool LTSKE::minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                            const arma::vec& v2, const arma::vec& v3) const {
  return minorIndicator(v0, v1, v2, v3, yPerm);
}

/*********************************
 * FullLexTauStarKernelEvaluator
 *********************************/

FLTSKE::FullLexTauStarKernelEvaluator(int xDim, int yDim): xDim(xDim), yDim(yDim) {
  arma::umat xPerms = permutations(xDim);
  arma::umat yPerms = permutations(yDim);

  for (int i = 0; i < xPerms.n_rows; i++) {
    for (int j = 0; j < yPerms.n_rows; j++) {
      LexTauStarKernelEvaluator a(xDim, yDim, xPerms.row(i).t(), yPerms.row(j).t());
      evaluators.push_back(a);
    }
  }
}

double FLTSKE::eval(const arma::mat& X, const arma::mat& Y) const {
  double sum = 0;
  for (int i = 0; i < evaluators.size(); i++) {
    sum += evaluators[i].eval(X, Y);
  }
  return sum;
}

int FLTSKE::order() const { return ord; }

/*********************************
 * PartialTauStarEvaluator
 *********************************/

PTSE::PartialTauStarEvaluator(int xDim, int yDim): xDim(xDim), yDim(yDim),
lowerBaseX(xDim), lowerBaseY(yDim), upperBaseX(xDim), upperBaseY(yDim) {
  lowerBaseX.fill(0);
  lowerBaseY.fill(0);
  upperBaseX.fill(std::numeric_limits<unsigned int>::max());
  upperBaseY.fill(std::numeric_limits<unsigned int>::max());
}

double PTSE::countGreaterEqOrLesserEqInXY(
    const arma::uvec& x,
    const arma::uvec& y,
    std::shared_ptr<OrthogonalRangeQuerier> orq,
    const bool& greaterInX, const bool& greaterInY) const {
  arma::uvec lower(xDim + yDim);
  arma::uvec upper(xDim + yDim);

  for (int i = 0; i < xDim; i++) {
    if (greaterInX) {
      lower(i) = x(i);
      upper(i) = std::numeric_limits<unsigned int>::max();
    } else {
      lower(i) = std::numeric_limits<unsigned int>::lowest();
      upper(i) = x(i);
    }
  }

  for (int i = 0; i < yDim; i++) {
    if (greaterInY) {
      lower(xDim + i) = y(i);
      upper(xDim + i) = std::numeric_limits<unsigned int>::max();
    } else {
      lower(xDim + i) = std::numeric_limits<unsigned int>::lowest();
      upper(xDim + i) = y(i);
    }
  }

  return orq->countInRange(lower, upper);
}

double PTSE::posConCount(const arma::uvec& x0, const arma::uvec& x1,
                         const arma::uvec& y0, const arma::uvec& y1,
                         std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  double val = 0;
  for (int i1 = 0; i1 < 2; i1++) {
    for (int i2 = 0; i2 < 2; i2++) {
      for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
          arma::uvec x = upperBaseX;
          arma::uvec y = upperBaseY;
          if (i1 == 1) {
            y = arma::min(y, y0);
          }
          if (i2 == 1) {
            y = arma::min(y, y1);
          }
          if (i3 == 1) {
            x = arma::min(x, x0);
          }
          if (i4 == 1) {
            x = arma::min(x, x1);
          }
          val += std::pow(-1, i1 + i2 + i3 + i4) *
            countGreaterEqOrLesserEqInXY(x, y, orq, false, false);
        }
      }
    }
  }
  return 2.0 * choose2(val);
}

double PTSE::negConCount(const arma::uvec& x0, const arma::uvec& x1,
                         const arma::uvec& y0, const arma::uvec& y1,
                         std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  double val = 0;
  for (int i1 = 0; i1 < 2; i1++) {
    for (int i2 = 0; i2 < 2; i2++) {
      for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
          arma::uvec x = upperBaseX;
          arma::uvec y = lowerBaseY;

          if (i1 == 1) {
            y = arma::max(y, y0);
          }
          if (i2 == 1) {
            y = arma::max(y, y1);
          }
          if (i3 == 1) {
            x = arma::min(x, x0);
          }
          if (i4 == 1) {
            x = arma::min(x, x1);
          }
          val += std::pow(-1, i1 + i2 + i3 + i4) *
            countGreaterEqOrLesserEqInXY(x, y, orq, false, true);
        }
      }
    }
  }
  return 2.0 * choose2(val);
}

arma::umat vecOfVecsToMat(const std::vector<std::vector<unsigned int> >& vecOfVecs) {
  if (vecOfVecs.size() == 0) {
    throw std::logic_error("Cannot construct a matrix from an empty vec of vecs");
  }
  int nrows = vecOfVecs.size();
  int ncols = vecOfVecs[0].size();
  arma::umat M = arma::zeros<arma::umat>(nrows, ncols);
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      M(i,j) = vecOfVecs[i][j];
    }
  }
  return M;
}

arma::uvec glue(const arma::uvec& v0, const arma::uvec& v1, const arma::uvec& v2,
               const arma::uvec& v3) {
  return arma::join_cols(arma::join_cols(v0, v1), arma::join_cols(v2, v3));
}

std::vector<unsigned int> toVector(const arma::uvec& v) {
  return arma::conv_to<std::vector<unsigned int> >::from(v);
}

std::shared_ptr<OrthogonalRangeQuerier> PTSE::createComparableOrq(const arma::umat& X,
                                                 const arma::umat& Y) const {
  std::vector<std::vector<unsigned int> > comparablePairs;
  int n = X.n_rows;

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::uvec x0 = X.row(i).t();
      arma::uvec x1 = X.row(j).t();
      arma::uvec y0 = Y.row(i).t();
      arma::uvec y1 = Y.row(j).t();

      if (all(y0 <= y1)) {
        comparablePairs.push_back(toVector(glue(x0, x1, y0, y1)));
      }
      if (all(y1 <= y0)) {
        comparablePairs.push_back(toVector(glue(x1, x0, y1, y0)));
      }
    }
  }
  if (comparablePairs.size() != 0) {
    return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(vecOfVecsToMat(comparablePairs)));
  } else {
    return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(arma::zeros<arma::umat>(0, 2 * (xDim + yDim))));
  }
}

std::shared_ptr<OrthogonalRangeQuerier> PTSE::createPairsOrq(const arma::umat& X,
                                            const arma::umat& Y) const {
  int n = X.n_rows;
  arma::umat pairs = arma::zeros<arma::umat>(n * n, 2 * (xDim + yDim));

  int k = 0;
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::uvec x0 = X.row(i).t();
      arma::uvec x1 = X.row(j).t();
      arma::uvec y0 = Y.row(i).t();
      arma::uvec y1 = Y.row(j).t();

      pairs.row(k) = glue(x0, x1, y0, y1).t();
      k++;
      pairs.row(k) = glue(x1, x0, y1, y0).t();
      k++;
    }
  }
  pairs.resize(k, 2 * (xDim + yDim));
  return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(pairs));
}

double PTSE::disCount(const arma::uvec& x0, const arma::uvec& x1,
                      const arma::uvec& y0, const arma::uvec& y1,
                      std::shared_ptr<OrthogonalRangeQuerier> compOrq,
                      std::shared_ptr<OrthogonalRangeQuerier> pairsOrq) const {
  double count = 0;
  for (int i1 = 0; i1 < 2; i1++) {
    for (int i2 = 0; i2 < 2; i2++) {
      for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
          for (int i5 = 0; i5 < 2; i5++) {
            for (int i6 = 0; i6 < 2; i6++) {
              arma::uvec lowerX0 = lowerBaseX, lowerX1 = lowerBaseX;
              arma::uvec lowerY0 = lowerBaseY, lowerY1 = lowerBaseY;
              arma::uvec upperX0 = upperBaseX, upperX1 = upperBaseX;
              arma::uvec upperY0 = upperBaseY, upperY1 = upperBaseY;

              if (i1 == 1) {
                upperX0 = arma::min(upperX0, x0);
              }

              if (i2 == 1) {
                upperX0 = arma::min(upperX0, x1);
              }

              if (i3 == 1) {
                upperX1 = arma::min(upperX1, x0);
              }

              if (i4 == 1) {
                upperX1 = arma::min(upperX1, x1);
              }

              if (i5 == 1) {
                upperY0 = arma::min(upperY0, y0);
              }

              if (i6 == 1) {
                lowerY1 = arma::max(lowerY1, y1);
              }

              count += std::pow(-1, i1 + i2 + i3 + i4 + i5 + i6) *
                pairsOrq->countInRange(toVector(glue(lowerX0, lowerX1, lowerY0, lowerY1)),
                                toVector(glue(upperX0, upperX1, upperY0, upperY1)));

              count += std::pow(-1, i1 + i2 + i3 + i4 + i5 + i6 + 1) *
                compOrq->countInRange(toVector(glue(lowerX0, lowerX1, lowerY0, lowerY1)),
                                     toVector(glue(upperX0, upperX1, upperY0, upperY1)));
            }
          }
        }
      }
    }
  }
  return count;
}

double PTSE::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::umat xJointRanks = toJointRankMatrix(X);
  arma::umat yJointRanks = toJointRankMatrix(Y);
  arma::umat allJointRanks = arma::join_rows(xJointRanks, yJointRanks);
  std::shared_ptr<OrthogonalRangeQuerier> orq =
    std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(allJointRanks));
  std::shared_ptr<OrthogonalRangeQuerier> compOrq = createComparableOrq(xJointRanks, yJointRanks);
  std::shared_ptr<OrthogonalRangeQuerier> pairsOrq = createPairsOrq(xJointRanks, yJointRanks);

  int numSamples = xJointRanks.n_rows;

  double sum = 0;
  for(int i = 0; i < numSamples - 1; i++) {
    for(int j = i + 1; j < numSamples; j++) {
      arma::uvec x0 = xJointRanks.row(i).t();
      arma::uvec x1 = xJointRanks.row(j).t();
      arma::uvec y0 = yJointRanks.row(i).t();
      arma::uvec y1 = yJointRanks.row(j).t();

      sum += 2.0 * posConCount(x0, x1, y0, y1, orq);
      sum += 2.0 * negConCount(x0, x1, y0, y1, orq);

      if (any(y0 < y1)) {
        sum -= 2.0 * disCount(x0, x1, y0, y1, compOrq, pairsOrq);
      }

      if (any(y1 < y0)) {
        sum -= 2.0 * disCount(x1, x0, y1, y0, compOrq, pairsOrq);
      }
    }
  }

  return sum / (nChooseM(1.0 * numSamples, 4.0) * 24);
}

/*********************************
 * JointTauStarKernelEvaluator
 *********************************/

JTSKE::JointTauStarKernelEvaluator(const arma::uvec& xOnOffVec,
                                   const arma::uvec& yOnOffVec):
  GenericTauStarKernelEvaluator(xOnOffVec.size(), yOnOffVec.size()),
  xOnOffVec(xOnOffVec), yOnOffVec(yOnOffVec) {
  for (int i = 0; i < xOnOffVec.size(); i++) {
    if (xOnOffVec(i) != 0.0 && xOnOffVec(i) != 1.0) {
      throw std::logic_error("Joint tau* requires a 0-1 valued vector as input");
    }
  }
  for (int i = 0; i < yOnOffVec.size(); i++) {
    if (yOnOffVec(i) != 0.0 && yOnOffVec(i) != 1.0) {
      throw std::logic_error("Joint tau* requires a 0-1 valued vector as input");
    }
  }
  if (!any(xOnOffVec == 1.0) || !any(yOnOffVec == 1.0)) {
    throw std::logic_error("Joint tau* input vectors must have >= one 1 each.");
  }
}

bool JTSKE::minorIndicatorX(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const {
  return minorIndicator(v0, v1, v2, v3, xOnOffVec);
}

bool JTSKE::minorIndicatorY(const arma::vec& v0, const arma::vec& v1,
                       const arma::vec& v2, const arma::vec& v3) const {
  return minorIndicator(v0, v1, v2, v3, yOnOffVec);
}

bool JTSKE::minorIndicator(const arma::vec& v0, const arma::vec& v1,
                           const arma::vec& v2, const arma::vec& v3,
                           const arma::uvec& onOffVec) const {
  for (int i = 0; i < onOffVec.size(); i++) {
      if ((onOffVec(i) == 0 && !(std::max(v2(i), v3(i)) <= std::min(v0(i), v1(i)))) ||
          (onOffVec(i) == 1 && !(std::min(v2(i), v3(i)) > std::max(v0(i), v1(i))))) {
      return false;
    }
  }
  return true;
}

/*********************************
 * JointTauStarEvaluator
 *********************************/

JTSE::JointTauStarEvaluator(const arma::uvec& xOnOffVec,
                            const arma::uvec& yOnOffVec): xOnOffVec(xOnOffVec),
                            yOnOffVec(yOnOffVec), xDim(xOnOffVec.size()),
                            yDim(yOnOffVec.size()) {}

bool JTSE::lessInPartialOrder(const arma::uvec& v0,
                              const arma::uvec& v1,
                              const arma::uvec& onOffVec) {
  for (int i = 0; i < v0.size(); i++) {
    if ((onOffVec(i) == 0 && !(v1(i) <= v0(i))) ||
        (onOffVec(i) == 1 && !(v0(i) < v1(i)))) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<OrthogonalRangeQuerier> JTSE::createComparableOrq(
    const arma::umat& X,
    const arma::umat& Y) const {
  std::vector<std::vector<unsigned int> > comparablePairs;
  int n = X.n_rows;

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::uvec x0 = X.row(i).t();
      arma::uvec x1 = X.row(j).t();
      arma::uvec y0 = Y.row(i).t();
      arma::uvec y1 = Y.row(j).t();

      if (any(y0 != y1)) {
        if (lessInPartialOrder(y0, y1, yOnOffVec)) {
          comparablePairs.push_back(toVector(glue(x0, x1, y0, y1)));
        } else if (lessInPartialOrder(y1, y0, yOnOffVec)) {
          comparablePairs.push_back(toVector(glue(x1, x0, y1, y0)));
        }
      }
    }
  }

  if (comparablePairs.size() != 0) {
    return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(vecOfVecsToMat(comparablePairs)));
  } else {
    return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(arma::zeros<arma::umat>(0, 2 * (xDim + yDim))));
  }
}

double JTSE::posConCount(
    const arma::uvec& x0, const arma::uvec& x1,
    const arma::uvec& y0, const arma::uvec& y1,
    std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  arma::uvec lower(xDim + yDim);
  arma::uvec upper(xDim + yDim);

  for (int i = 0; i < xDim; i++) {
    if (xOnOffVec(i) == 0) {
      unsigned int maxVal = std::max(x0(i), x1(i));
      lower(i) = maxVal;
      upper(i) = std::numeric_limits<unsigned int>::max();
    } else {
      unsigned int minVal = std::min(x0(i), x1(i));
      lower(i) = std::numeric_limits<unsigned int>::lowest();
      upper(i) = minVal - 1; // Minus 1 since not including upper
    }
  }

  for (int i = 0; i < yDim; i++) {
    if (yOnOffVec(i) == 0) {
      unsigned int maxVal = std::max(y0(i), y1(i));
      lower(i + xDim) = maxVal;
      upper(i + xDim) = std::numeric_limits<unsigned int>::max();
    } else {
      unsigned int minVal = std::min(y0(i), y1(i));
      lower(i + xDim) = std::numeric_limits<unsigned int>::lowest();
      upper(i + xDim) = minVal - 1; //Minus 1 since not including upper
    }
  }

  return 2.0 * choose2(orq->countInRange(lower, upper));
}

double JTSE::negConCount(const arma::uvec& x0, const arma::uvec& x1,
                         const arma::uvec& y0, const arma::uvec& y1,
                         std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  arma::uvec lower(xDim + yDim);
  arma::uvec upper(xDim + yDim);

  for (int i = 0; i < xDim; i++) {
    if (xOnOffVec(i) == 0) {
      unsigned int maxVal = std::max(x0(i), x1(i));
      lower(i) = maxVal;
      upper(i) = std::numeric_limits<unsigned int>::max();
    } else {
      unsigned int minVal = std::min(x0(i), x1(i));
      lower(i) = std::numeric_limits<unsigned int>::lowest();
      upper(i) = minVal - 1; //Minus 1 since not including upper
    }
  }

  for (int i = 0; i < yDim; i++) {
    if (yOnOffVec(i) == 0) {
      unsigned int minVal = std::min(y0(i), y1(i));
      lower(i + xDim) = std::numeric_limits<unsigned int>::lowest();
      upper(i + xDim) = minVal;
    } else {
      unsigned int maxVal = std::max(y0(i), y1(i));
      lower(i + xDim) = maxVal + 1; // Plus 1 since not including lower
      upper(i + xDim) = std::numeric_limits<unsigned int>::max();
    }
  }

  return 2.0 * choose2(orq->countInRange(lower, upper));
}

double JTSE::disCount(const arma::uvec& x0, const arma::uvec& x1,
                      const arma::uvec& y0, const arma::uvec& y1,
                      std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  arma::uvec lower(2 * (xDim + yDim));
  arma::uvec upper(2 * (xDim + yDim));

  for (int i = 0; i < xDim; i++) {
    if (xOnOffVec(i) == 0) {
      unsigned int maxVal = std::max(x0(i), x1(i));
      lower(i) = maxVal;
      upper(i) = std::numeric_limits<unsigned int>::max();
    } else {
      unsigned int minVal = std::min(x0(i), x1(i));
      lower(i) = std::numeric_limits<unsigned int>::lowest();
      upper(i) = minVal - 1; // Minus 1 since not including upper
    }
    // Duplicating for the other x
    lower(i + xDim) = lower(i);
    upper(i + xDim) = upper(i);
  }

  for (int i = 0; i < yDim; i++) {
    if (yOnOffVec(i) == 0) {
      lower(i + 2 * xDim) = y1(i);
      upper(i + 2 * xDim) = std::numeric_limits<unsigned int>::max();
    } else {
      lower(i + 2 * xDim) = std::numeric_limits<unsigned int>::lowest();
      upper(i + 2 * xDim) = y1(i) - 1; // Minus 1 since not including upper
    }
  }

  for (int i = 0; i < yDim; i++) {
    if (yOnOffVec(i) == 0) {
      lower(i + 2 * xDim + yDim) = std::numeric_limits<unsigned int>::lowest();
      upper(i + 2 * xDim + yDim) = y0(i);
    } else {
      lower(i + 2 * xDim + yDim) = y0(i) + 1; // Plus 1 since not including lower
      upper(i + 2 * xDim + yDim) = std::numeric_limits<unsigned int>::max();
    }
  }

  return orq->countInRange(lower, upper);
}

double JTSE::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::umat xJointRanks = toJointRankMatrix(X);
  arma::umat yJointRanks = toJointRankMatrix(Y);
  arma::umat allJointRanks = arma::join_rows(xJointRanks, yJointRanks);
  std::shared_ptr<OrthogonalRangeQuerier> orq =
    std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(allJointRanks));
  //auto start = std::chrono::steady_clock::now();
  std::shared_ptr<OrthogonalRangeQuerier> compOrq = createComparableOrq(xJointRanks, yJointRanks);
  //auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(std::chrono::steady_clock::now() - start);
  //std::cout << duration.count() / 1000.0 << std::endl;

  int numSamples = X.n_rows;

  double sum = 0;
  for(int i = 0; i < numSamples - 1; i++) {
    for(int j = i + 1; j < numSamples; j++) {
      arma::uvec x0 = xJointRanks.row(i).t();
      arma::uvec x1 = xJointRanks.row(j).t();
      arma::uvec y0 = yJointRanks.row(i).t();
      arma::uvec y1 = yJointRanks.row(j).t();

      sum += 2.0 * posConCount(x0, x1, y0, y1, orq);
      sum += 2.0 * negConCount(x0, x1, y0, y1, orq);
      if (lessInPartialOrder(y0, y1, yOnOffVec)) {
        sum += -2.0 * disCount(x0, x1, y0, y1, compOrq);
      } else if (lessInPartialOrder(y1, y0, yOnOffVec)) {
        sum += -2.0 * disCount(x1, x0, y1, y0, compOrq);
      }
    }
  }
  return sum / (nChooseM(1.0 * numSamples, 4.0) * 24);
}







