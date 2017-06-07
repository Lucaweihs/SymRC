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

#include "OrthogonalRangeQuerier.h"
#include "HelperFunctions.h"

typedef OrthogonalRangeTensor ort;
typedef AlignedRangeTree art;
typedef NaiveRangeCounter nrc;

/***
 * NaiveRangeCounter
 ***/

nrc::NaiveRangeCounter(const arma::umat& jointRanks): jointRanksTranspose(jointRanks.t()) {}

unsigned int nrc::countInRange(const arma::uvec& lower,
                               const arma::uvec& upper) const {
  if (size() == 0) {
    return 0;
  }
  unsigned int count = 0;
  for (int i = 0; i < jointRanksTranspose.n_cols; i++) {
    if (arma::all(lower <= jointRanksTranspose.col(i)) &&
        arma::all(jointRanksTranspose.col(i) <= upper)) {
      count++;
    }
  }
  return count;
}

int nrc::size() const {
  return jointRanksTranspose.n_cols;
}

std::shared_ptr<OrthogonalRangeQuerier>
  NaiveRangeCounterBuilder::build(const arma::umat& jointRanks) const {
  return std::shared_ptr<OrthogonalRangeQuerier>(new NaiveRangeCounter(jointRanks));
}

/***
 * AlignedRangeTree
 ***/

art::AlignedRangeTree(const arma::umat& jointRanks): numPoints(jointRanks.n_rows) {
  std::vector<RangeTree::Point<unsigned int,bool> > points;
  for (int i = 0; i < jointRanks.n_rows; i++) {
    std::vector<unsigned int> position = arma::conv_to<std::vector<unsigned int> >::from(jointRanks.row(i));
    RangeTree::Point<unsigned int,bool> point(position, true);
    points.push_back(point);
  }
  if (points.size() != 0) {
    rtree = std::shared_ptr<RangeTree::RangeTree<unsigned int,bool> >(
      new RangeTree::RangeTree<unsigned int,bool>(points));
  }
}

unsigned int art::countInRange(const arma::uvec& lower,
                               const arma::uvec& upper) const {
  if (size() == 0) {
    return 0;
  }
  return rtree->countInRange(arma::conv_to<std::vector<unsigned int> >::from(lower),
                             arma::conv_to<std::vector<unsigned int> >::from(upper));
}

int art::size() const {
  return numPoints;
}

std::shared_ptr<OrthogonalRangeQuerier>
  AlignedRangeTreeBuilder::build(const arma::umat& jointRanks) const {
    return std::shared_ptr<OrthogonalRangeQuerier>(new AlignedRangeTree(jointRanks));
}

/***************************
 * OrthogonalRangeTensor
 ***************************/

ort::OrthogonalRangeTensor(const arma::umat& jointRanks):
  dims(jointRanks.n_cols) {
  zeroOneMat = powerSetMat(jointRanks.n_cols);
  for (int i = 0; i < zeroOneMat.n_rows; i++) {
    zeroOneMatParity.push_back(arma::accu(zeroOneMat.row(i)) % 2 == 1);
  }
  if (arma::any(arma::vectorise(jointRanks) == 0)) {
    throw Rcpp::exception("Joint rank matrix input to OrthogonalRangeTensor"
                            " must have no 0s");
  }
  int numTensorEntries = 1;
  for (int i = 0; i < jointRanks.n_cols; i++) {
    dims(i) = jointRanks.col(i).max() + 1;
    numTensorEntries *= dims(i);
  }
  tensorAsVec = arma::zeros<arma::uvec>(numTensorEntries);
  arma::uvec visited = arma::zeros<arma::uvec>(numTensorEntries);

  for (int i = 0; i < jointRanks.n_rows; i++) {
    unsigned int yar = indexToInt(jointRanks.row(i).t());
    tensorAsVec(yar) += 1;
  }
  createTensorRecurse(dims - 1, visited);
}

unsigned int ort::indexToInt(const arma::uvec& index) const {
  unsigned int intIndex = 0;
  unsigned int prodSoFar = 1;
  for (int i = 0; i < dims.size(); i++) {
    intIndex += index(i) * prodSoFar;
    prodSoFar *= dims(i);
  }
  return intIndex;
}

unsigned int ort::createTensorRecurse(const arma::uvec& index, arma::uvec& visited) {
  if (arma::any(index == 0)) {
    return 0;
  }
  unsigned int intInd = indexToInt(index);
  if (visited(intInd) == 0) {
    unsigned int posPart = tensorAsVec(indexToInt(index));
    unsigned int negPart = 0;
    for (int i = 1; i < zeroOneMat.n_rows; i++) {
      if (zeroOneMatParity[i]) {
        posPart += createTensorRecurse(index - zeroOneMat.row(i).t(), visited);
      } else {
        negPart += createTensorRecurse(index - zeroOneMat.row(i).t(), visited);
      }
    }
    if (posPart < negPart) {
      throw Rcpp::exception("posPart should never be less than negPart in createTensorRecurse");
    }
    tensorAsVec(intInd) = posPart - negPart;
    visited(intInd) = 1;
  }
  return tensorAsVec(intInd);
}

unsigned int ort::countInRange(const arma::uvec& lower,
                               const arma::uvec& upper) const {
  arma::uvec tmpLower(lower.size());
  arma::uvec tmpUpper(upper.size());
  if (arma::any(upper < lower)) {
    return 0;
  }
  for (int i = 0; i < tmpLower.size(); i++) {
    if (lower(i) != 0) {
      tmpLower(i) = lower(i) - 1;
    } else {
      tmpLower(i) = 0;
    }

    if (upper(i) >= dims(i)) {
      tmpUpper(i) = dims(i) - 1;
    } else {
      tmpUpper(i) = upper(i);
    }
  }
  unsigned int posPart = 0;
  unsigned int negPart = 0;

  arma::uvec index(dims);
  for (int i = 0; i < zeroOneMat.n_rows; i++) {
    for (int j = 0; j < tmpLower.size(); j++) {
      if (zeroOneMat(i,j) == 1) {
        index(j) = tmpLower(j);
      } else {
        index(j) = tmpUpper(j);
      }
    }

    unsigned int indexAsInt = indexToInt(index);
    if (!zeroOneMatParity[i]) {
      posPart += tensorAsVec(indexAsInt);
    } else {
      negPart += tensorAsVec(indexAsInt);
    }
  }
  if (posPart < negPart) {
    throw Rcpp::exception("posPart should never be less than negPart in countInRange");
  }
  return posPart - negPart;
}

int ort::size() const {
  return tensorAsVec(indexToInt(dims - 1));
}

int ort::getDimAtIndex(unsigned int i) const {
  return dims(i);
}

std::shared_ptr<OrthogonalRangeQuerier>
  OrthogonalRangeTensorBuilder::build(const arma::umat& jointRanks) const {
    return std::shared_ptr<OrthogonalRangeQuerier>(new OrthogonalRangeTensor(jointRanks));
}
