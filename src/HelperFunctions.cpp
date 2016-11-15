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

#include "HelperFunctions.h"
#include <cmath>

// [[Rcpp::export]]
arma::uvec intersectSorted(const arma::uvec& vec1, const arma::uvec& vec2) {
  int initialSize = (vec1.size() < vec2.size()) ? vec1.size() : vec2.size();
  arma::uvec intersectedVec = arma::uvec(initialSize, arma::fill::zeros);

  int interVecInd = 0;
  int vec1Ind = 0;
  int vec2Ind = 0;
  while (vec1Ind < vec1.size() && vec2Ind < vec2.size()) {
    if (vec1(vec1Ind) == vec2(vec2Ind)) {
      intersectedVec[interVecInd] = vec1(vec1Ind);
      interVecInd++;
      vec1Ind++;
      vec2Ind++;
    } else if(vec1(vec1Ind) < vec2(vec2Ind)) {
      vec1Ind++;
    } else {
      vec2Ind++;
    }
  }
  intersectedVec.resize(interVecInd);
  return intersectedVec;
}

// [[Rcpp::export]]
arma::uvec unionSorted(const arma::uvec& vec1, const arma::uvec& vec2) {
  std::vector<unsigned int> unionV;
  if (vec1.size() != 0 && vec2.size() != 0) {
    unionV.push_back(std::min(vec1(0), vec2(0)));
  } else if (vec1.size() != 0) {
    unionV.push_back(vec1(0));
  } else if (vec2.size() != 0) {
    unionV.push_back(vec2(0));
  }

  int vec1Ind = 0;
  int vec2Ind = 0;

  while (vec1Ind < vec1.size() && vec2Ind < vec2.size()) {
    double vec1Val = vec1(vec1Ind);
    double vec2Val = vec2(vec2Ind);
    if (vec1Val < vec2Val) {
      if (vec1Val != unionV[unionV.size() - 1]) {
        unionV.push_back(vec1Val);
      }
      vec1Ind++;
    } else if (vec2Val < vec1Val) {
      if (vec2Val != unionV[unionV.size() - 1]) {
        unionV.push_back(vec2Val);
      }
      vec2Ind++;
    } else {
      if (vec1Val != unionV[unionV.size() - 1]) {
        unionV.push_back(vec1Val);
      }
      vec1Ind++;
      vec2Ind++;
    }
  }

  if (vec1Ind < vec1.size()) {
    for (int i = vec1Ind; i < vec1.size(); i++) {
      if (vec1[i] != unionV[unionV.size() - 1]) {
        unionV.push_back(vec1[i]);
      }
    }
  }

  if (vec2Ind < vec2.size()) {
    for (int i = vec2Ind; i < vec2.size(); i++) {
      if (vec2[i] != unionV[unionV.size() - 1]) {
        unionV.push_back(vec2[i]);
      }
    }
  }
  return arma::conv_to<arma::uvec>::from(unionV);
}

// [[Rcpp::export]]
arma::uvec setDiffSorted(const arma::uvec& vec1, const arma::uvec& vec2) {
  if (vec1.size() == 0 || vec2.size() == 0) {
    return vec1;
  }
  arma::uvec setDiffVec = arma::uvec(vec1.size(), arma::fill::zeros);

  int setDiffVecInd = 0;
  int vec1Ind = 0;
  int vec2Ind = 0;
  while(vec1Ind < vec1.size()) {
    if (vec1[vec1Ind] == vec2[vec2Ind]) {
      vec1Ind++;
      vec2Ind++;
    } else if (vec1[vec1Ind] < vec2[vec2Ind] || vec2Ind >= vec2.size()) {
      setDiffVec[setDiffVecInd] = vec1[vec1Ind];
      setDiffVecInd++;
      vec1Ind++;
    } else {
      vec2Ind++;
    }
  }
  setDiffVec.resize(setDiffVecInd);
  return setDiffVec;
}

// [[Rcpp::export]]
arma::uvec complementSorted(const arma::uvec& inds, int maxInd) {
  if (inds.size() > maxInd + 1) {
    Rcpp::stop("Size of vector given to complementSorted is > maxInd + 1.");
  }
  arma::uvec complementInds = arma::uvec(maxInd + 1 - inds.size());
  int k = 0;
  int l = 0;
  for (int i = 0; i <= maxInd; i++) {
    if (k < inds.size() && inds(k) == i) {
      k++;
    } else {
      complementInds(l) = i;
      l++;
    }
  }

  return complementInds;
}

// [[Rcpp::export]]
arma::uvec intToUVec(unsigned int uint, int length) {
  if (length > 16) {
    Rcpp::stop("intToVec only works with positive length <= 16.");
  }
  arma::uvec vec(length);
  for (int i = 0; i < length; i++) {
    vec(i) = uint % 2;
    uint = (uint - uint % 2) / 2;
  }
  return vec;
}

// [[Rcpp::export]]
int zeroOneVecToInt(const arma::uvec& zeroOneVec) {
  int val = 0;
  int factor = 1;
  for (int i = 0; i < zeroOneVec.size(); i++) {
    val += zeroOneVec[i] * factor;
    factor *= 2;
  }
  return val;
}

// [[Rcpp::export]]
unsigned int intPow(int base, int exponent) {
  if (base < 0 || exponent < 0) {
    Rcpp::stop("Base and exponent in intPow must be >=0.");
  }
  if (exponent == 0) {
    return 1;
  }
  if (base == 0) {
    return 0;
  }
  if (std::log(std::numeric_limits<unsigned int>::max()) / std::log(1.0 * base)
        <= exponent) {
    Rcpp::stop("intPow computation likely to overflow, stopping to prevent this.");
  }
  unsigned int val = 1;
  while (exponent > 0) {
    val *= base;
    exponent--;
  }
  return val;
}

int nChooseM(int n, int m) {
  if (n < 0 || m < 0) {
    Rcpp::stop("Invalid input to nChooseM.");
  }
  if (m > n) {
    return 0;
  }
  int val = 1;
  for (int i = m + 1; i <= n; i++) {
    val *= i;
  }
  for (int i = 2; i <= (n - m); i++) {
    val /= i;
  }
  return val;
}

double nChooseM(double n, double m) {
  if (n < 0 || m < 0) {
    Rcpp::stop("Invalid input to nChooseM.");
  }
  if (m > n) {
    return 0.0;
  }
  double val = 0.0;
  for (int i = m + 1; i <= n; i++) {
    val += std::log(1.0 * i);
  }
  for (int i = 2; i <= (n - m); i++) {
    val -= std::log(1.0 * i);
  }
  return std::exp(val);
}

int factorial(int n) {
  if (n < 0) {
    throw std::domain_error("Factorial of negative numbers is not defined.");
  }
  if (n == 0 || n == 1) {
    return 1;
  }
  return n * factorial(n - 1);
}

// [[Rcpp::export]]
arma::umat permutations(int n) {
  arma::umat allPerms = arma::zeros<arma::umat>(factorial(n), n);
  if (n == 0 || n == 1) {
    return allPerms;
  }
  arma::umat otherPerms = permutations(n - 1);
  for (int bigInd = 0; bigInd < n; bigInd++) {
    for (int i = 0; i < otherPerms.n_rows; i++) {
      for (int j = 0; j < n; j++) {
        int position = n - 1 - bigInd;
        int rowInd = bigInd * otherPerms.n_rows + i;
        if (j == position) {
          allPerms(rowInd, j) = n - 1;
        } else if(j > position) {
          allPerms(rowInd, j) = otherPerms(i, j - 1);
        } else {
          allPerms(rowInd, j) = otherPerms(i, j);
        }
      }
    }
  }
  return allPerms;
}

// [[Rcpp::export]]
arma::mat orderStats(arma::mat M) {
  for (int i = 0; i < M.n_cols; i++) {
    M.col(i) = arma::sort(M.col(i));
  }
  return M;
}

void printVec(std::vector<double> vec) {
  if (vec.size() == 0) {
    std::cout << "empty" << std::endl;
    return;
  }
  for (int i = 0; i < vec.size() - 1; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << vec[vec.size() - 1] << std::endl;
}

void printVec(std::vector<int> vec) {
  if (vec.size() == 0) {
    std::cout << "empty" << std::endl;
    return;
  }
  for (int i = 0; i < vec.size() - 1; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << vec[vec.size() - 1] << std::endl;
}

void printVec(std::vector<bool> vec) {
  if (vec.size() == 0) {
    std::cout << "empty" << std::endl;
    return;
  }
  for (int i = 0; i < vec.size() - 1; i++) {
    std::cout << (vec[i] ? "T" : "F") << " ";
  }
  std::cout << (vec[vec.size() - 1] ? "T" : "F") << std::endl;
}
