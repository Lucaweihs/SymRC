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

typedef GenericTauStarKernelEvaluator GTSKE;
typedef PartialTauStarKernelEvaluator PTSKE;
typedef LexTauStarKernelEvaluator LTSKE;
typedef FullLexTauStarKernelEvaluator FLTSKE;

typedef PartialTauStarEvaluator PTSE;

const arma::umat GTSKE::perms = permutations(4);

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
    if(minorIndicatorY(y2, y3, y0, y1)) {
      fullSum += 1.0;
    }
    if(minorIndicatorY(y0, y2, y1, y3)) {
      fullSum += -2.0;
    }
  }
  return fullSum / perms.n_rows;
}

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

PTSE::PartialTauStarEvaluator(int xDim, int yDim): xDim(xDim), yDim(yDim) {}

double PTSE::countGreaterEqInX(const arma::vec& x,
                               const EmpiricalDistribution& ed) const {
  arma::vec y = arma::zeros<arma::vec>(yDim);
  for (int i = 0; i < y.size(); i++) {
    y(i) = std::numeric_limits<double>::lowest();
  }
  return countGreaterEqInXY(x, y, ed);
}

double PTSE::countGreaterEqInY(const arma::vec& y,
                               const EmpiricalDistribution& ed) const {
  arma::vec x = arma::zeros<arma::vec>(xDim);
  for (int i = 0; i < x.size(); i++) {
    x(i) = std::numeric_limits<double>::lowest();
  }
  return countGreaterEqInXY(x, y, ed);
}

double PTSE::countGreaterEqInXY(const arma::vec& x, const arma::vec& y,
                              const EmpiricalDistribution& ed) const {
  std::vector<double> lower, upper;
  std::vector<bool> withLower, withUpper;

  for (int i = 0; i < xDim; i++) {
    lower.push_back(x(i));
    upper.push_back(std::numeric_limits<double>::max());
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  for (int i = 0; i < yDim; i++) {
    lower.push_back(y(i));
    upper.push_back(std::numeric_limits<double>::max());
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  return ed.countInRange(lower, upper, withLower, withUpper);
}

double PTSE::countLesserEqInY(const arma::vec& y,
                              const EmpiricalDistribution& ed) const {
  arma::vec x = arma::zeros<arma::vec>(xDim);
  for (int i = 0; i < x.size(); i++) {
    x(i) = std::numeric_limits<double>::lowest();
  }
  return countGreaterEqXLesserEqY(x, y, ed);
}

double PTSE::countGreaterEqXLesserEqY(const arma::vec& x, const arma::vec& y,
                              const EmpiricalDistribution& ed) const {
  std::vector<double> lower, upper;
  std::vector<bool> withLower, withUpper;

  for (int i = 0; i < xDim; i++) {
    lower.push_back(x(i));
    upper.push_back(std::numeric_limits<double>::max());
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  for (int i = 0; i < yDim; i++) {
    lower.push_back(std::numeric_limits<double>::lowest());
    upper.push_back(y(i));
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  return ed.countInRange(lower, upper, withLower, withUpper);
}

double PTSE::posConCount(const arma::vec& x0, const arma::vec& x1,
                         const arma::vec& y0, const arma::vec& y1,
                         const EmpiricalDistribution& ed) const {
  arma::vec maxX01 = arma::max(x0, x1);
  arma::vec maxY01 = arma::max(y0, y1);

  double geqX0 = countGreaterEqInX(x0, ed);
  double geqX1 = countGreaterEqInX(x1, ed);
  double geqY0 = countGreaterEqInY(y0, ed);
  double geqY1 = countGreaterEqInY(y1, ed);

  double geqX01 = countGreaterEqInX(arma::max(x0, x1), ed);
  double geqY01 = countGreaterEqInY(maxY01, ed);

  double geqX0Y0 = countGreaterEqInXY(x0, y0, ed);
  double geqX0Y1 = countGreaterEqInXY(x0, y1, ed);
  double geqX1Y0 = countGreaterEqInXY(x1, y0, ed);
  double geqX1Y1 = countGreaterEqInXY(x1, y1, ed);

  double geqX01Y0 = countGreaterEqInXY(maxX01, y0, ed);
  double geqX01Y1 = countGreaterEqInXY(maxX01, y1, ed);
  double geqX0Y01 = countGreaterEqInXY(x0, maxY01, ed);
  double geqX1Y01 = countGreaterEqInXY(x1, maxY01, ed);

  double geqX01Y01 = countGreaterEqInXY(maxX01, maxY01, ed);

    return 2.0 * choose2(
      ed.size() - (
          (geqX0 + geqX1 + geqY0 + geqY1) -
            (geqX01 + geqY01 + geqX0Y0 + geqX0Y1 + geqX1Y0 + geqX1Y1) +
            (geqX01Y0 + geqX01Y1 + geqX0Y01 + geqX1Y01) -
            geqX01Y01)
    );
}

double PTSE::negConCount(const arma::vec& x0, const arma::vec& x1,
                         const arma::vec& y0, const arma::vec& y1,
                         const EmpiricalDistribution& ed) const {
  arma::vec maxX01 = arma::max(x0, x1);
  arma::vec minY01 = arma::min(y0, y1);

  double geqX0 = countGreaterEqInX(x0, ed);
  double geqX1 = countGreaterEqInX(x1, ed);
  double leqY0 = countLesserEqInY(y0, ed);
  double leqY1 = countLesserEqInY(y1, ed);

  double geqX01 = countGreaterEqInX(arma::max(x0, x1), ed);
  double leqY01 = countLesserEqInY(minY01, ed);

  double geqX0leqY0 = countGreaterEqXLesserEqY(x0, y0, ed);
  double geqX0leqY1 = countGreaterEqXLesserEqY(x0, y1, ed);
  double geqX1leqY0 = countGreaterEqXLesserEqY(x1, y0, ed);
  double geqX1leqY1 = countGreaterEqXLesserEqY(x1, y1, ed);

  double geqX01leqY0 = countGreaterEqXLesserEqY(maxX01, y0, ed);
  double geqX01leqY1 = countGreaterEqXLesserEqY(maxX01, y1, ed);
  double geqX0leqY01 = countGreaterEqXLesserEqY(x0, minY01, ed);
  double geqX1leqY01 = countGreaterEqXLesserEqY(x1, minY01, ed);

  double geqX01leqY01 = countGreaterEqXLesserEqY(maxX01, minY01, ed);

  return 2.0 * choose2(
      ed.size() - (
          (geqX0 + geqX1 + leqY0 + leqY1) -
            (geqX01 + leqY01 + geqX0leqY0 + geqX0leqY1 + geqX1leqY0 + geqX1leqY1) +
            (geqX01leqY0 + geqX01leqY1 + geqX0leqY01 + geqX1leqY01) -
            geqX01leqY01)
  );
}

arma::mat vecOfVecsToMat(const std::vector<std::vector<double> >& vecOfVecs) {
  if (vecOfVecs.size() == 0) {
    throw std::logic_error("Cannot construct a matrix from an empty vec of vecs");
  }
  int nrows = vecOfVecs.size();
  int ncols = vecOfVecs[0].size();
  arma::mat M = arma::zeros<arma::mat>(nrows, ncols);
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      M(i,j) = vecOfVecs[i][j];
    }
  }
  return M;
}

arma::vec glue(const arma::vec& v0, const arma::vec& v1, const arma::vec& v2,
               const arma::vec& v3) {
  return arma::join_cols(arma::join_cols(v0, v1), arma::join_cols(v2, v3));
}

std::vector<double> toVector(const arma::vec& v) {
  return arma::conv_to<std::vector<double> >::from(v);
}

EmpiricalDistribution PTSE::createComparableED(const arma::mat& X,
                                               const arma::mat& Y) const {
  std::vector<std::vector<double> > comparablePairs;
  int n = X.n_rows;

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec x0 = X.row(i).t();
      arma::vec x1 = X.row(j).t();
      arma::vec y0 = Y.row(i).t();
      arma::vec y1 = Y.row(j).t();

      if (any(y0 != y1)) {
        if (all(y0 <= y1)) {
          comparablePairs.push_back(toVector(glue(x0, x1, y0, y1)));
        } else if (all(y1 <= y0)) {
          comparablePairs.push_back(toVector(glue(x1, x0, y1, y0)));
        }
      }
    }
  }

  if (comparablePairs.size() != 0) {
    return EmpiricalDistribution(vecOfVecsToMat(comparablePairs));
  } else {
    return EmpiricalDistribution(arma::zeros<arma::mat>(0, 2 * (xDim + yDim)));
  }
}

EmpiricalDistribution PTSE::createIncomparableED(const arma::mat& X,
                                                 const arma::mat& Y) const {
  std::vector<std::vector<double> > incomparablePairs;
  int n = X.n_rows;

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec x0 = X.row(i).t();
      arma::vec x1 = X.row(j).t();
      arma::vec y0 = Y.row(i).t();
      arma::vec y1 = Y.row(j).t();

      if (!all(y0 <= y1) && !all(y1 <= y0)) {
        incomparablePairs.push_back(toVector(glue(x0, x1, y0, y1)));
        incomparablePairs.push_back(toVector(glue(x1, x0, y1, y0)));
      }
    }
  }
  if (incomparablePairs.size() != 0) {
    return EmpiricalDistribution(vecOfVecsToMat(incomparablePairs));
  } else {
    return EmpiricalDistribution(arma::zeros<arma::mat>(0, 2 * (xDim + yDim)));
  }
}

EmpiricalDistribution PTSE::createPairsED(const arma::mat& X,
                                          const arma::mat& Y) const {
  int n = X.n_rows;
  arma::mat pairs = arma::zeros<arma::mat>(n * n, 2 * (xDim + yDim));

  int k = 0;
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      arma::vec x0 = X.row(i).t();
      arma::vec x1 = X.row(j).t();
      arma::vec y0 = Y.row(i).t();
      arma::vec y1 = Y.row(j).t();

      if (!all(y1 <= y0)) {
        pairs.row(k) = glue(x0, x1, y0, y1).t();
        k++;
      }
      if (!all(y0 <= y1)) {
        pairs.row(k) = glue(x1, x0, y1, y0).t();
        k++;
      }
    }
  }
  pairs.resize(k, 2 * (xDim + yDim));
  return EmpiricalDistribution(pairs);
}

double PTSE::disCount(const arma::vec& x0, const arma::vec& x1,
                      const arma::vec& y0, const arma::vec& y1,
                      const EmpiricalDistribution& ed) const {
  std::vector<bool> withLower, withUpper;
  arma::vec lowerBaseX = arma::ones<arma::vec>(x0.size());
  arma::vec lowerBaseY = arma::ones<arma::vec>(y0.size());
  arma::vec upperBaseX = arma::ones<arma::vec>(x0.size());
  arma::vec upperBaseY = arma::ones<arma::vec>(y0.size());
  lowerBaseX *= std::numeric_limits<double>::lowest();
  lowerBaseY *= std::numeric_limits<double>::lowest();
  upperBaseX *= std::numeric_limits<double>::max();
  upperBaseY *= std::numeric_limits<double>::max();
  for (int i = 0; i < 2*(xDim + yDim); i++) {
    withLower.push_back(true);
    withUpper.push_back(true);
  }

  double count = 0;
  bool firstLoop = true;
  // std::cout << "\nSTARTING LOOP" << std::endl;
  for (int i1 = 0; i1 < 2; i1++) {
    for (int i2 = 0; i2 < 2; i2++) {
      for (int i3 = 0; i3 < 2; i3++) {
        for (int i4 = 0; i4 < 2; i4++) {
          for (int i5 = 0; i5 < 2; i5++) {
            for (int i6 = 0; i6 < 2; i6++) {
              for (int i7 = 0; i7 < 2; i7++) {
                for (int i8 = 0; i8 < 2; i8++) {
              if (firstLoop) {
                firstLoop = false;
                continue;
              }
                  arma::vec lowerX0 = lowerBaseX, lowerX1 = lowerBaseX;
                  arma::vec lowerY0 = lowerBaseY, lowerY1 = lowerBaseY;
                  arma::vec upperX0 = upperBaseX, upperX1 = upperBaseX;
                  arma::vec upperY0 = upperBaseY, upperY1 = upperBaseY;

                  if (i1 == 1) { // X3 <= X2
                    lowerX0 = arma::max(lowerX0, x0);
                  }

                  if (i2 == 1) { // X4 <= X2
                    lowerX0 = arma::max(lowerX0, x1);
                  }

                  if (i3 == 1) { // X3 <= X1
                    lowerX1 = arma::max(lowerX1, x0);
                  }

                  if (i4 == 1) { // X4 <= X1
                    lowerX1 = arma::max(lowerX1, x1);
                  }

                  if (i5 == 1) { // Y1 <= Y3
                    upperY0 = arma::min(upperY0, y0);
                  }

                  if (i6 == 1) { // Y4 <= Y1
                    lowerY0 = arma::max(lowerY0, y1);
                  }

                  if (i7 == 1) { // Y2 <= Y3
                    upperY1 = arma::min(upperY1, y0);
                  }

                  if (i8 == 1) { // Y4 <= Y2
                    lowerY1 = arma::max(lowerY1, y1);
                  }

                  double val = std::pow(-1, i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + 1) *
                    ed.countInRange(toVector(glue(lowerX0, lowerX1, lowerY0, lowerY1)),
                                    toVector(glue(upperX0, upperX1, upperY0, upperY1)),
                                    withLower, withUpper);
                  count += val;

                  // if (val != 0) {
                  //   std::cout << "Points" << std::endl;
                  //   x0.t().print();
                  //   x1.t().print();
                  //   y0.t().print();
                  //   y1.t().print();
                  //
                  //   std::cout << "Lowers" << std::endl;
                  //
                  //   lowerX0.t().print();
                  //   lowerX1.t().print();
                  //   lowerY0.t().print();
                  //   lowerY1.t().print();
                  //
                  //   std::cout << "Uppers" << std::endl;
                  //   upperX0.t().print();
                  //   upperX1.t().print();
                  //   upperY0.t().print();
                  //   upperY1.t().print();
                  //
                  //   std::cout << "withLower and withUpper" << std::endl;
                  //   printVec(withLower);
                  //   printVec(withUpper);
                  //   std::cout << "(signed) Points in subregion: " << val << std::endl;
                  // }
                }
              }
            }
          }
        }
      }
    }
  }
  // std::cout << "Pairs in empirical distribution: " << std::endl;
  // ed.getSamples().print();
  // std::cout << "\nTotal # pairs: " <<ed.size() << ", pairs in region: " << count << std::endl;
  // std::cout << "ENDING LOOP" << std::endl;

  return ed.size() - count;
}

double PTSE::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::mat allSamples = arma::join_rows(X, Y);
  EmpiricalDistribution ed(allSamples);
  EmpiricalDistribution pairsEd = createPairsED(X, Y);
  int numSamples = X.n_rows;

  double sum = 0;
  for(int i = 0; i < numSamples - 1; i++) {
    for(int j = i + 1; j < numSamples; j++) {
      arma::vec x0 = X.row(i).t();
      arma::vec x1 = X.row(j).t();
      arma::vec y0 = Y.row(i).t();
      arma::vec y1 = Y.row(j).t();

      sum += 2.0 * posConCount(x0, x1, y0, y1, ed);
      sum += 2.0 * negConCount(x0, x1, y0, y1, ed);

      if (any(y0 < y1)) {
        double leqY0 = countLesserEqInY(y0, ed);
        double geqY1 = countGreaterEqInY(y1, ed);

        double geqX0 = countGreaterEqInX(x0, ed);
        double geqX1 = countGreaterEqInX(x1, ed);
        double geqX01 = countGreaterEqInX(arma::max(x0, x1), ed);
        double geqX0leqY0 = countGreaterEqXLesserEqY(x0, y0, ed);
        double geqX1leqY0 = countGreaterEqXLesserEqY(x1, y0, ed);
        double geqX01leqY0 = countGreaterEqXLesserEqY(arma::max(x0, x1), y0, ed);

        double geqX0Y1 = countGreaterEqInXY(x0, y1, ed);
        double geqX1Y1 = countGreaterEqInXY(x1, y1, ed);
        double geqX01Y1 = countGreaterEqInXY(arma::max(x0, x1), y1, ed);

        int countCorrectX = ed.size() - geqX0 - geqX1 + geqX01;

        int countLessEq0 = leqY0 - geqX0leqY0 - geqX1leqY0 + geqX01leqY0;
        int countGreater1 = geqY1 - geqX0Y1 - geqX1Y1 + geqX01Y1;
        int countGreater0 = countCorrectX - countLessEq0;
        int countMiddle = countCorrectX - countLessEq0 - countGreater1;

        int a = countLessEq0 * countGreater0;
        int b = countMiddle * countGreater1;
        int c = disCount(x0, x1, y0, y1, pairsEd);
        sum += -2 * (a + b + c);
      }

      if (any(y1 < y0)) {
        arma::vec tmp;
        tmp = y0; y0 = y1; y1 = tmp;
        tmp = x0; x0 = x1; x1 = tmp;
        double leqY0 = countLesserEqInY(y0, ed);
        double geqY1 = countGreaterEqInY(y1, ed);

        double geqX0 = countGreaterEqInX(x0, ed);
        double geqX1 = countGreaterEqInX(x1, ed);
        double geqX01 = countGreaterEqInX(arma::max(x0, x1), ed);
        double geqX0leqY0 = countGreaterEqXLesserEqY(x0, y0, ed);
        double geqX1leqY0 = countGreaterEqXLesserEqY(x1, y0, ed);
        double geqX01leqY0 = countGreaterEqXLesserEqY(arma::max(x0, x1), y0, ed);

        double geqX0Y1 = countGreaterEqInXY(x0, y1, ed);
        double geqX1Y1 = countGreaterEqInXY(x1, y1, ed);
        double geqX01Y1 = countGreaterEqInXY(arma::max(x0, x1), y1, ed);

        int countCorrectX = ed.size() - geqX0 - geqX1 + geqX01;

        int countLessEq0 = leqY0 - geqX0leqY0 - geqX1leqY0 + geqX01leqY0;
        int countGreater1 = geqY1 - geqX0Y1 - geqX1Y1 + geqX01Y1;
        int countGreater0 = countCorrectX - countLessEq0;
        int countMiddle = countCorrectX - countLessEq0 - countGreater1;

        int a = countLessEq0 * countGreater0;
        int b = countMiddle * countGreater1;
        int c = disCount(x0, x1, y0, y1, pairsEd);
        sum += -2 * (a + b + c);
      }
    }
  }

  return sum / (nChooseM(1.0 * numSamples, 4.0) * 24);
}

