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
#include "Hoeffding.h"
#include "HelperFunctions.h"

typedef HoeffdingDEvaluator hde;
typedef HoeffdingREvaluator hre;
typedef HoeffdingDKernelEvaluator hdke;
typedef HoeffdingRKernelEvaluator hrke;

const arma::umat hoeffDPosPerms = {{0,1,2,3,4}, {3,2,1,0,4}};
const arma::umat hoeffDNegPerms = {{3,1,2,0,4}, {0,2,1,3,4}};

hdke::HoeffdingDKernelEvaluator(int xDim, int yDim):
  SymRCKernelEvaluator(xDim, yDim, hoeffDPosPerms, hoeffDNegPerms) {}

bool hdke::minorIndicatorX(const arma::mat& vecs) const {
  return arma::all(vecs.row(0) <= vecs.row(4)) &&
    arma::all(vecs.row(1) <= vecs.row(4)) &&
    !arma::all(vecs.row(2) <= vecs.row(4)) &&
    !arma::all(vecs.row(3) <= vecs.row(4));
}

bool hdke::minorIndicatorY(const arma::mat& vecs) const {
  return minorIndicatorX(vecs);
}

hde::HoeffdingDEvaluator(
  int xDim, int yDim, std::shared_ptr<OrthogonalRangeQuerierBuilder> orqb):
  xDim(xDim), yDim(yDim), lowerBaseX(xDim), lowerBaseY(yDim), upperBaseX(xDim),
  upperBaseY(yDim), orqBuilder(orqb) {
  lowerBaseX.fill(0);
  lowerBaseY.fill(0);
  upperBaseX.fill(std::numeric_limits<unsigned int>::max());
  upperBaseY.fill(std::numeric_limits<unsigned int>::max());
}

double hde::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::umat xJointRanks = toJointRankMatrix(X);
  arma::umat yJointRanks = toJointRankMatrix(Y);
  arma::umat allJointRanks = arma::join_rows(xJointRanks, yJointRanks);
  std::shared_ptr<OrthogonalRangeQuerier> orq = orqBuilder->build(allJointRanks);
  int n = xJointRanks.n_rows;

  double sum = 0;
  for(int i = 0; i < n ; i++) {
    arma::uvec x = xJointRanks.row(i).t();
    arma::uvec y = yJointRanks.row(i).t();

    unsigned int x0y0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                              arma::join_cols(x, y));
    unsigned int y0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                        arma::join_cols(upperBaseX, y));
    unsigned int x0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                        arma::join_cols(x, upperBaseY));

    double c00 = x0y0 - 1;
    double c10 = y0 - x0y0;
    double c01 = x0 - x0y0;
    double c11 = n - x0 - y0 + x0y0;

    sum += 4 * (choose2(c00) * choose2(c11) + choose2(c10) * choose2(c01)) -
      2 * c00 * c01 * c10 * c11;
  }
  return sum / (1.0 * n * (n - 1) * (n - 2) * (n - 3) * (n - 4));
}

hre::HoeffdingREvaluator(
  int xDim, int yDim, std::shared_ptr<OrthogonalRangeQuerierBuilder> orqb):
  xDim(xDim), yDim(yDim), lowerBaseX(xDim), lowerBaseY(yDim), upperBaseX(xDim),
  upperBaseY(yDim), orqBuilder(orqb) {
  lowerBaseX.fill(0);
  lowerBaseY.fill(0);
  upperBaseX.fill(std::numeric_limits<unsigned int>::max());
  upperBaseY.fill(std::numeric_limits<unsigned int>::max());
  perms = permutations(xDim + yDim);
}

double hre::evalLoop(int dim,
                     arma::uvec& index,
                     const arma::umat& X,
                     const arma::umat& Y,
                     std::shared_ptr<OrthogonalRangeQuerier> orq) const {
  double sum = 0.0;
  int offset = 0;
  if (dim != 0) {
    offset = index(dim - 1) + 1;
  }

  if (dim != index.size() - 1) {

    for (int i = offset; i < X.n_rows - (index.size() - (dim + 1)); i++) {
      index(dim) = i;
      sum += evalLoop(dim + 1, index, X, Y, orq);
    }
  } else {
    arma::uvec x(xDim);
    arma::uvec y(yDim);
    for (int i = offset; i < X.n_rows; i++) {
      index(dim) = i;
      for (int j = 0; j < perms.n_rows; j++) {
        arma::uvec permIndex = index.elem(perms.row(j));
        for (int k = 0; k < xDim; k++) {
          x(k) = X(permIndex(k), k);
        }
        for (int k = 0; k < yDim; k++) {
          y(k) = Y(permIndex(xDim + k), k);
        }
        unsigned int x0y0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                              arma::join_cols(x, y));
        unsigned int y0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                            arma::join_cols(upperBaseX, y));
        unsigned int x0 = orq->countInRange(arma::join_cols(lowerBaseX, lowerBaseY),
                                            arma::join_cols(x, upperBaseY));

        double c00 = x0y0;
        double c10 = y0 - x0y0;
        double c01 = x0 - x0y0;
        double c11 = X.n_rows - x0 - y0 + x0y0;

        for (int k = 0; k < index.size(); k++) {
          bool isXLess = arma::all(X.row(index(k)).t() <= x);
          bool isYLess = arma::all(Y.row(index(k)).t() <= y);
          c00 -= (isXLess && isYLess) ? 1 : 0;
          c10 -= (!isXLess && isYLess) ? 1 : 0;
          c01 -= (isXLess && !isYLess) ? 1 : 0;
          c11 -= (!isXLess && !isYLess) ? 1 : 0;
        }

        sum += 4 * (choose2(c00) * choose2(c11) + choose2(c10) * choose2(c01)) -
          2 * c00 * c01 * c10 * c11;
      }
    }
  }
  return sum;
}

double hre::eval(const arma::mat& X, const arma::mat& Y) const {
  arma::umat xJointRanks = toJointRankMatrix(X);
  arma::umat yJointRanks = toJointRankMatrix(Y);
  arma::umat allJointRanks = arma::join_rows(xJointRanks, yJointRanks);
  std::shared_ptr<OrthogonalRangeQuerier> orq = orqBuilder->build(allJointRanks);
  int n = xJointRanks.n_rows;

  arma::uvec index = arma::zeros<arma::uvec>(xDim + yDim);

  double sum = evalLoop(0, index, xJointRanks, yJointRanks, orq);
  for (int i = 0; i < 4 + xDim + yDim; i++) {
    sum /= n - i;
  }
  return sum;
}

arma::umat createHoeffdingRPosPerms(int dim) {
  arma::uvec ending(dim - 1);
  for (int i = 5; i < 4 + dim; i++) {
    ending(i - 5) = i;
  }
  return arma::join_rows(hoeffDPosPerms,
                         arma::join_cols(ending.t(), ending.t()));
}

arma::umat createHoeffdingRNegPerms(int dim) {
  arma::uvec ending(dim - 1);
  for (int i = 5; i < 4 + dim; i++) {
    ending(i - 5) = i;
  }
  return arma::join_rows(hoeffDNegPerms,
                         arma::join_cols(ending.t(), ending.t()));
}

hrke::HoeffdingRKernelEvaluator(int xDim, int yDim):
  SymRCKernelEvaluator(xDim, yDim, createHoeffdingRPosPerms(xDim + yDim),
                       createHoeffdingRNegPerms(xDim + yDim)) {}

bool hrke::minorIndicatorX(const arma::mat& vecs) const {
  arma::rowvec toCompare(xDim);
  for (int i = 0; i < xDim; i++) {
    toCompare(i) = vecs(4 + i, i);
  }
  return arma::all(vecs.row(0) <= toCompare) &&
    arma::all(vecs.row(1) <= toCompare) &&
    !arma::all(vecs.row(2) <= toCompare) &&
    !arma::all(vecs.row(3) <= toCompare);
}

bool hrke::minorIndicatorY(const arma::mat& vecs) const {
  arma::rowvec toCompare(yDim);
  for (int i = 0; i < yDim; i++) {
    toCompare(i) = vecs(4 + xDim + i, i);
  }
  return arma::all(vecs.row(0) <= toCompare) &&
    arma::all(vecs.row(1) <= toCompare) &&
    !arma::all(vecs.row(2) <= toCompare) &&
    !arma::all(vecs.row(3) <= toCompare);
}
