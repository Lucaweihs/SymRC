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

#ifndef WCM_HelperFunctions
#define WCM_HelperFunctions

// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

arma::uvec intersectSorted(const arma::uvec& vec1, const arma::uvec& vec2);
arma::uvec unionSorted(const arma::uvec& vec1, const arma::uvec& vec2);
arma::uvec setDiffSorted(const arma::uvec& vec1, const arma::uvec& vec2);
arma::uvec complementSorted(const arma::uvec& inds, int maxInd);
arma::uvec intToUVec(unsigned int uint, int length);
int zeroOneVecToInt(const arma::uvec& zeroOneVec);
unsigned int intPow(int base, int exponent);
int nChooseM(int n, int m);
double nChooseM(double n, double m);
double choose2(double n);
arma::umat permutations(int n);
arma::mat orderStats(arma::mat M);

void printVec(std::vector<double> vec);
void printVec(std::vector<int> vec);
void printVec(std::vector<bool> vec);

#endif
