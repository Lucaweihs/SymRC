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

#include "IntegratedMinor.h"
#include "HelperFunctions.h"

typedef IntegratedMinorKernelEvaluator IMKE;
typedef IntegratedMinorEvaluator IME;
typedef SubsetMinorPartition SMP;

SMP::SubsetMinorPartition(int dim, arma::uvec leftInds, arma::uvec rightInds) {
  inds0 = arma::unique(leftInds);
  inds1 = arma::unique(rightInds);

  if (dim <= inds0.max() || dim <= inds1.max()) {
    throw std::logic_error("Dimension must be > all input indices.");
  }

  allInds = unionSorted(inds0, inds1);
  intersection = intersectSorted(inds0, inds1);
  complement = complementSorted(allInds, dim);
  counts = arma::zeros<arma::uvec>(dim);
  counts(inds0) += 1;
  counts(inds1) += 1;

  symmetricDiff = arma::join_cols(inds0Unique, inds1Unique);

  inds0Unique = setDiffSorted(inds0, intersection);
  inds1Unique = setDiffSorted(inds1, intersection);
}
const arma::uvec& SMP::getAllInds() const { return allInds; }
const arma::uvec& SMP::getNonUniqueLeftInds() const { return inds0; }
const arma::uvec& SMP::getNonUniqueRightInds() const { return inds1; }
const arma::uvec& SMP::getUniqueLeftInds() const { return inds0Unique; }
const arma::uvec& SMP::getUniqueRightInds() const { return inds1Unique; }
const arma::uvec& SMP::getIntersection() const { return intersection; }
const arma::uvec& SMP::getCounts() const { return counts; }
const arma::uvec& SMP::getComplement() const { return complement; }
const arma::uvec& SMP::getSymmetricDiff() const { return symmetricDiff; }

IMKE::IntegratedMinorKernelEvaluator(int xDim, int yDim,
                              arma::uvec xInds0, arma::uvec xInds1,
                              arma::uvec yInds0, arma::uvec yInds1) :
  xPart(xDim, xInds0, xInds1), yPart(yDim, yInds0, yInds1),
  perms(permutations(ord)) { }

int IMKE::order() const {
  return ord;
}

bool IMKE::minorIndicator(const arma::vec& v0, const arma::vec& v1,
                         const arma::vec& v2, const arma::vec& v3,
                         const SubsetMinorPartition& part) const {
  arma::vec xMax01 = arma::max(v0, v1);
  arma::vec xMax012 = arma::max(xMax01, v2);
  arma::vec xMax013 = arma::max(xMax01, v3);
  arma::vec xMin23 = arma::min(v2, v3);

  const arma::uvec& inter = part.getIntersection();;
  if (any(xMin23.elem(inter) <= xMax01.elem(inter))) {
    return false;
  }

  const arma::uvec& inds0 = part.getUniqueLeftInds();
  const arma::uvec& inds1 = part.getUniqueRightInds();
  if ((any(v2.elem(inds0) <= xMax013.elem(inds0)) ||
      any(v3.elem(inds1) <= xMax012.elem(inds1))) &&
      (any(v2.elem(inds1) <= xMax013.elem(inds1)) ||
      any(v3.elem(inds0) <= xMax012.elem(inds0)))) {
    return false;
  }

  return true;
}

bool IMKE::weightIndicator(const arma::mat& V, const SubsetMinorPartition& part) const {
  // std::cout << "Lin space\n";
  // arma::linspace<arma::uvec>(0, 3, 4).print();
  // std::cout << "V\n";
  // V.print();
  // std::cout << "V first four\n";
  // V.rows(arma::linspace<arma::uvec>(0, 3, 4)).print();

  arma::mat orderStatsFirstFour =
    orderStats(V.rows(arma::linspace<arma::uvec>(0, 3, 4)));
  // std::cout << "Order stats\n";
  // orderStatsFirstFour.print();

  const arma::uvec& counts = part.getCounts();
  for (int i = 0; i < counts.size(); i++) {
    if (counts(i) == 0) {
      if (orderStatsFirstFour(3, i) > V(4, i)) {
        return false;
      }
    } else if (counts(i) == 1) {
      if (orderStatsFirstFour(2, i) > V(4, i) ||
          V(4, i) >= orderStatsFirstFour(3, i)) {
        return false;
      }
    } else {
      if (orderStatsFirstFour(1, i) > V(4, i) ||
          V(4, i) >= orderStatsFirstFour(2, i)) {
        return false;
      }
    }
  }
  return true;
}

double IMKE::eval(const arma::mat& X, const arma::mat& Y) const {
  double fullSum = 0;
  for (int i = 0; i < perms.n_rows; i++) {
    arma::vec x0 = X.row(perms(i,0)).t();
    arma::vec x1 = X.row(perms(i,1)).t();
    arma::vec x2 = X.row(perms(i,2)).t();
    arma::vec x3 = X.row(perms(i,3)).t();
    arma::vec x4 = X.row(perms(i,4)).t();

    if (!minorIndicator(x0, x1, x2, x3, xPart)) {
      continue;
    }

    arma::vec y0 = Y.row(perms(i,0)).t();
    arma::vec y1 = Y.row(perms(i,1)).t();
    arma::vec y2 = Y.row(perms(i,2)).t();
    arma::vec y3 = Y.row(perms(i,3)).t();
    arma::vec y4 = Y.row(perms(i,4)).t();

    double val = 0;
    if (minorIndicator(y0, y1, y2, y3, yPart)) {
      val = 1.0;
    } else if(minorIndicator(y2, y3, y0, y1, yPart)) {
      val = 1.0;
    } else if(minorIndicator(y0, y2, y1, y3, yPart)) {
      val = -2.0;
    }

    if (weightIndicator(X.rows(perms.row(i)), xPart) &&
        weightIndicator(Y.rows(perms.row(i)), yPart)) {
      fullSum += val;
    }
  }
  return fullSum / perms.n_rows;
}

IME::IntegratedMinorEvaluator(int xDim, int yDim,
                              arma::uvec xInds0, arma::uvec xInds1,
                              arma::uvec yInds0, arma::uvec yInds1):
  xPart(xDim, xInds0, xInds1), yPart(yDim, yInds0, yInds1), xDim(xDim),
  yDim(yDim) {
  xIndsEq = (xPart.getNonUniqueLeftInds().size() == xPart.getNonUniqueRightInds().size()) &&
    all(xPart.getNonUniqueLeftInds() == xPart.getNonUniqueRightInds());
  yIndsEq = (yPart.getNonUniqueLeftInds().size() == yPart.getNonUniqueRightInds().size()) &&
    all(yPart.getNonUniqueLeftInds() == yPart.getNonUniqueRightInds());

  for (int i = 0; i < xDim + yDim; i++) {
    minLower.push_back(std::numeric_limits<double>::lowest());
  }
}

double choose2(double x) {
  return nChooseM(x, 2.0);
}

double IME::countGreaterInX(const std::vector<double> point,
                            const arma::uvec& xGreaterInds,
                            const EmpiricalDistribution& ed) const {
  arma::uvec yGreaterInds = arma::zeros<arma::uvec>(0);
  return countGreaterInXY(point, xGreaterInds, yGreaterInds, ed);
}

double IME::countGreaterInY(const std::vector<double> point,
                            const arma::uvec& yGreaterInds,
                            const EmpiricalDistribution& ed) const {
  arma::uvec xGreaterInds = arma::zeros<arma::uvec>(0);
  return countGreaterInXY(point, xGreaterInds, yGreaterInds, ed);
}

double IME::countGreaterInXY(const std::vector<double> point,
                             const arma::uvec& xGreaterInds,
                             const arma::uvec& yGreaterInds,
                             const EmpiricalDistribution& ed) const {
  std::vector<bool> withLower, withUpper;

  for (int i = 0; i < xDim + yDim; i++) {
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  std::vector<double> lower = minLower;
  std::vector<double> upper = point;

  for (int i = 0; i < xGreaterInds.size(); i++) {
    int ind = xGreaterInds(i);
    withLower[ind] = false;
    lower[ind] = point[ind];
    upper[ind] = std::numeric_limits<double>::max();
  }
  for (int i = 0; i < yGreaterInds.size(); i++) {
    int ind = yGreaterInds(i);
    withLower[ind + xDim] = false;
    lower[ind + xDim] = point[ind + xDim];
    upper[ind + xDim] = std::numeric_limits<double>::max();
  }
  return ed.countInRange(lower, upper, withLower, withUpper);
}

double posConSum(double leqCount, double grCount00,
                             double grCount11, bool xIndsEq, bool yIndsEq) {
  double toReturn = (2 * choose2(leqCount - 1));
  if (xIndsEq && yIndsEq) {
    toReturn *= 2 * choose2(grCount00);
  } else {
    toReturn *= grCount00 * grCount11;
  }
  return toReturn;
}

double negConSum(double grXCount0, double grXCount1,
                 double grYCount0, double grYCount1,
                 bool xIndsEq, bool yIndsEq) {
  double toReturn = 1;
  if (xIndsEq) {
    toReturn *= 2 * choose2(grXCount0);
  } else {
    toReturn *= grXCount0 * grXCount1;
  }
  if (yIndsEq) {
    toReturn *= 2 * choose2(grYCount0);
  } else {
    toReturn *= grYCount0 * grYCount1;
  }
  return toReturn;
}

double disSum(double leqCount, double grXCount0,
              double grYCount0, double grXYCount11) {
  return -2.0 * (1.0 * (leqCount - 1)) * grYCount0 * grXCount0 * grXYCount11;
}

double IME::eval(const arma::mat& X, const arma::mat& Y) const {
  if (xDim != X.n_cols || yDim != Y.n_cols) {
    throw std::logic_error("Dimensions of X and Y do not match initially given"
      "dimensions.");
  }
  arma::mat allSamples = arma::join_rows(X, Y);
  EmpiricalDistribution ed(allSamples);

  const arma::uvec& xInds0 = xPart.getNonUniqueLeftInds();
  const arma::uvec& xInds1 = xPart.getNonUniqueRightInds();
  const arma::uvec& yInds0 = yPart.getNonUniqueLeftInds();
  const arma::uvec& yInds1 = yPart.getNonUniqueRightInds();

  double val = 0;
  for (int i = 0; i < X.n_rows; i++) {
    arma::vec x = X.row(i).t();
    arma::vec y = Y.row(i).t();

    std::vector<double> point = arma::conv_to<std::vector<double> >::from(x);
    std::vector<double> yVec = arma::conv_to<std::vector<double> >::from(y);
    point.insert(point.end(), y.begin(), y.end());

    arma::uvec empty = arma::zeros<arma::uvec>(0);
    double leqXYCount = countGreaterInXY(point, empty, empty, ed);
    double grXCount0 = countGreaterInX(point, xInds0, ed);
    double grXCount1 = countGreaterInX(point, xInds1, ed);
    double grYCount0 = countGreaterInY(point, yInds0, ed);
    double grYCount1 = countGreaterInY(point, yInds1, ed);

    double grXYCount00 = countGreaterInXY(point, xInds0, yInds0, ed);
    double grXYCount10 = countGreaterInXY(point, xInds1, yInds0, ed);
    double grXYCount01 = countGreaterInXY(point, xInds0, yInds1, ed);
    double grXYCount11 = countGreaterInXY(point, xInds1, yInds1, ed);

    val += posConSum(leqXYCount, grXYCount00, grXYCount11, xIndsEq, yIndsEq);
    val += negConSum(grXCount0, grXCount1, grYCount0, grYCount1, xIndsEq, yIndsEq);
    val += disSum(leqXYCount, grXCount0, grYCount0, grXYCount11);

    if (!xIndsEq) {
      val += posConSum(leqXYCount, grXYCount10, grXYCount01, xIndsEq, yIndsEq);
      val += negConSum(grXCount1, grXCount0, grYCount0, grYCount1, xIndsEq, yIndsEq);
      val += disSum(leqXYCount, grXCount1, grYCount0, grXYCount01);
    }

    if (!yIndsEq) {
      val += posConSum(leqXYCount, grXYCount01, grXYCount10, xIndsEq, yIndsEq);
      val += negConSum(grXCount0, grXCount1, grYCount1, grYCount0, xIndsEq, yIndsEq);
      val += disSum(leqXYCount, grXCount0, grYCount1, grXYCount10);
    }

    if (!xIndsEq && !yIndsEq) {
      val += posConSum(leqXYCount, grXYCount11, grXYCount00, xIndsEq, yIndsEq);
      val += negConSum(grXCount1, grXCount0, grYCount1, grYCount0, xIndsEq, yIndsEq);
      val += disSum(leqXYCount, grXCount1, grYCount1, grXYCount00);
    }
  }

  int n = X.n_rows;
  return val / (120 * nChooseM(1.0 * n, 1.0 * 5));
}
