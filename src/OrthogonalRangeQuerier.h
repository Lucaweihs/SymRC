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

#ifndef SymRC_OrthogonalRangeQuerier
#define SymRC_OrthogonalRangeQuerier

#include "RcppArmadillo.h"
#include "RangeTree.h"

class OrthogonalRangeQuerier {
public:
  virtual unsigned int countInRange(const arma::uvec& lower,
                                    const arma::uvec& upper) const = 0;
  virtual int size() const = 0;
};

class AlignedRangeTree : public OrthogonalRangeQuerier {
private:
  std::vector<bool> withLower;
  std::vector<bool> withUpper;
  const unsigned int numPoints;
  std::shared_ptr<RangeTree::RangeTree<unsigned int,bool> > rtree;

public:
  AlignedRangeTree(const arma::umat& jointRanks);
  unsigned int countInRange(const arma::uvec& lower,
                            const arma::uvec& upper) const;
  int size() const;
};

class OrthogonalRangeTensor : public OrthogonalRangeQuerier {
private:
  arma::uvec dims;
  arma::uvec tensorAsVec;
  arma::umat zeroOneMat;
  std::vector<unsigned int> lower;
  std::vector<unsigned int> upper;

  unsigned int indexToInt(const arma::uvec& index) const;
  unsigned int createTensorRecurse(const arma::uvec& index, arma::uvec& visited);
//  unsigned int indexToInt(const arma::uvec& index);

public:
  OrthogonalRangeTensor(const arma::umat& jointRanks);
  unsigned int countInRange(const arma::uvec& lower,
                            const arma::uvec& upper) const;
  int size() const;
  int getDimAtIndex(unsigned int index) const;
};

#endif
