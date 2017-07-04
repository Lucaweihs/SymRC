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

#ifndef SymRC_Hoeffding
#define SymRC_Hoeffding

#include "NaiveUStatistics.h"
#include "OrthogonalRangeQuerier.h"

class HoeffdingDKernelEvaluator : public SymRCKernelEvaluator {
private:
  bool minorIndicatorX(const arma::mat& vecs) const;
  bool minorIndicatorY(const arma::mat& vecs) const;

public:
  HoeffdingDKernelEvaluator(int xDim, int yDim);
};

class HoeffdingDEvaluator {
private:
  arma::uvec lowerBaseX;
  arma::uvec lowerBaseY;
  arma::uvec upperBaseX;
  arma::uvec upperBaseY;
  std::shared_ptr<OrthogonalRangeQuerierBuilder> orqBuilder;

public:
  HoeffdingDEvaluator(int xDim, int yDim,
                      std::shared_ptr<OrthogonalRangeQuerierBuilder> orqb);
  double eval(const arma::mat& X, const arma::mat& Y) const;
};

class HoeffdingREvaluator {
private:
  int xDim, yDim;
  arma::uvec lowerBaseX;
  arma::uvec lowerBaseY;
  arma::uvec upperBaseX;
  arma::uvec upperBaseY;
  arma::umat perms;
  std::shared_ptr<OrthogonalRangeQuerierBuilder> orqBuilder;

public:
  HoeffdingREvaluator(int xDim, int yDim,
                      std::shared_ptr<OrthogonalRangeQuerierBuilder> orqb);
  double eval(const arma::mat& X, const arma::mat& Y) const;
  double evalLoop(int dim,
                  arma::uvec& index,
                  const arma::umat& X,
                  const arma::umat& Y,
                  const std::shared_ptr<OrthogonalRangeQuerier>& orq,
                  unsigned int& iters) const;
};

class HoeffdingRKernelEvaluator : public SymRCKernelEvaluator {
private:
  bool minorIndicatorX(const arma::mat& vecs) const;
  bool minorIndicatorY(const arma::mat& vecs) const;

public:
  HoeffdingRKernelEvaluator(int xDim, int yDim);
};

#endif