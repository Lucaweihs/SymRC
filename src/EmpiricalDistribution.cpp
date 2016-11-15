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

#include "EmpiricalDistribution.h"

typedef EmpiricalDistribution ed;

ed::EmpiricalDistribution(arma::mat samples) : samples(samples) {
  std::vector<RangeTree::Point<int> > points;
  for (int i = 0; i < samples.n_rows; i++) {
    std::vector<double> position = arma::conv_to<std::vector<double> >::from(samples.row(i));
    RangeTree::Point<int> point(position, 0);
    points.push_back(point);
  }
  rtree = std::shared_ptr<RangeTree::RangeTree<int> >(new RangeTree::RangeTree<int>(points));
}

int ed::countInRange(const std::vector<double>& lower,
                 const std::vector<double>& upper,
                 const std::vector<bool>& withLower,
                 const std::vector<bool>& withUpper) const {
  return rtree->countInRange(lower, upper, withLower, withUpper);
}

double ed::probOfRange(const std::vector<double>& lower,
                   const std::vector<double>& upper,
                   const std::vector<bool>& withLower,
                   const std::vector<bool>& withUpper) const {
  return (1.0 * countInRange(lower, upper, withLower, withUpper)) /
    samples.n_rows;
}