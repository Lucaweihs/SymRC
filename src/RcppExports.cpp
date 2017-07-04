// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// powerSetMat
arma::umat powerSetMat(unsigned int n);
RcppExport SEXP SymRC_powerSetMat(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(powerSetMat(n));
    return rcpp_result_gen;
END_RCPP
}
// toJointRankMatrix
arma::umat toJointRankMatrix(const arma::mat& samples);
RcppExport SEXP SymRC_toJointRankMatrix(SEXP samplesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type samples(samplesSEXP);
    rcpp_result_gen = Rcpp::wrap(toJointRankMatrix(samples));
    return rcpp_result_gen;
END_RCPP
}
// intersectSorted
arma::uvec intersectSorted(const arma::uvec& vec1, const arma::uvec& vec2);
RcppExport SEXP SymRC_intersectSorted(SEXP vec1SEXP, SEXP vec2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec1(vec1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec2(vec2SEXP);
    rcpp_result_gen = Rcpp::wrap(intersectSorted(vec1, vec2));
    return rcpp_result_gen;
END_RCPP
}
// unionSorted
arma::uvec unionSorted(const arma::uvec& vec1, const arma::uvec& vec2);
RcppExport SEXP SymRC_unionSorted(SEXP vec1SEXP, SEXP vec2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec1(vec1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec2(vec2SEXP);
    rcpp_result_gen = Rcpp::wrap(unionSorted(vec1, vec2));
    return rcpp_result_gen;
END_RCPP
}
// setDiffSorted
arma::uvec setDiffSorted(const arma::uvec& vec1, const arma::uvec& vec2);
RcppExport SEXP SymRC_setDiffSorted(SEXP vec1SEXP, SEXP vec2SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec1(vec1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type vec2(vec2SEXP);
    rcpp_result_gen = Rcpp::wrap(setDiffSorted(vec1, vec2));
    return rcpp_result_gen;
END_RCPP
}
// complementSorted
arma::uvec complementSorted(const arma::uvec& inds, int maxInd);
RcppExport SEXP SymRC_complementSorted(SEXP indsSEXP, SEXP maxIndSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type inds(indsSEXP);
    Rcpp::traits::input_parameter< int >::type maxInd(maxIndSEXP);
    rcpp_result_gen = Rcpp::wrap(complementSorted(inds, maxInd));
    return rcpp_result_gen;
END_RCPP
}
// intToUVec
arma::uvec intToUVec(unsigned int uint, int length);
RcppExport SEXP SymRC_intToUVec(SEXP uintSEXP, SEXP lengthSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type uint(uintSEXP);
    Rcpp::traits::input_parameter< int >::type length(lengthSEXP);
    rcpp_result_gen = Rcpp::wrap(intToUVec(uint, length));
    return rcpp_result_gen;
END_RCPP
}
// zeroOneVecToInt
int zeroOneVecToInt(const arma::uvec& zeroOneVec);
RcppExport SEXP SymRC_zeroOneVecToInt(SEXP zeroOneVecSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::uvec& >::type zeroOneVec(zeroOneVecSEXP);
    rcpp_result_gen = Rcpp::wrap(zeroOneVecToInt(zeroOneVec));
    return rcpp_result_gen;
END_RCPP
}
// intPow
unsigned int intPow(int base, int exponent);
RcppExport SEXP SymRC_intPow(SEXP baseSEXP, SEXP exponentSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type base(baseSEXP);
    Rcpp::traits::input_parameter< int >::type exponent(exponentSEXP);
    rcpp_result_gen = Rcpp::wrap(intPow(base, exponent));
    return rcpp_result_gen;
END_RCPP
}
// permutations
arma::umat permutations(int n);
RcppExport SEXP SymRC_permutations(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(permutations(n));
    return rcpp_result_gen;
END_RCPP
}
// orderStats
arma::mat orderStats(arma::mat M);
RcppExport SEXP SymRC_orderStats(SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(orderStats(M));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingROrthTensor
double hoeffdingROrthTensor(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingROrthTensor(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingROrthTensor(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingRNaive
double hoeffdingRNaive(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingRNaive(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingRNaive(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingRFromDef
double hoeffdingRFromDef(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingRFromDef(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingRFromDef(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingDRangeTree
double hoeffdingDRangeTree(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingDRangeTree(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingDRangeTree(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingDNaive
double hoeffdingDNaive(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingDNaive(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingDNaive(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// hoeffdingDFromDef
double hoeffdingDFromDef(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_hoeffdingDFromDef(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(hoeffdingDFromDef(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// jointTauStarRangeTree
double jointTauStarRangeTree(const arma::mat& X, const arma::mat& Y, const arma::uvec& xOnOffVec, const arma::uvec& yOnOffVec);
RcppExport SEXP SymRC_jointTauStarRangeTree(SEXP XSEXP, SEXP YSEXP, SEXP xOnOffVecSEXP, SEXP yOnOffVecSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xOnOffVec(xOnOffVecSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yOnOffVec(yOnOffVecSEXP);
    rcpp_result_gen = Rcpp::wrap(jointTauStarRangeTree(X, Y, xOnOffVec, yOnOffVec));
    return rcpp_result_gen;
END_RCPP
}
// jointTauStarNaive
double jointTauStarNaive(const arma::mat& X, const arma::mat& Y, const arma::uvec& xOnOffVec, const arma::uvec& yOnOffVec);
RcppExport SEXP SymRC_jointTauStarNaive(SEXP XSEXP, SEXP YSEXP, SEXP xOnOffVecSEXP, SEXP yOnOffVecSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xOnOffVec(xOnOffVecSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yOnOffVec(yOnOffVecSEXP);
    rcpp_result_gen = Rcpp::wrap(jointTauStarNaive(X, Y, xOnOffVec, yOnOffVec));
    return rcpp_result_gen;
END_RCPP
}
// jointTauStarFromDef
double jointTauStarFromDef(const arma::mat& X, const arma::mat& Y, const arma::uvec& xOnOffVec, const arma::uvec& yOnOffVec);
RcppExport SEXP SymRC_jointTauStarFromDef(SEXP XSEXP, SEXP YSEXP, SEXP xOnOffVecSEXP, SEXP yOnOffVecSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xOnOffVec(xOnOffVecSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yOnOffVec(yOnOffVecSEXP);
    rcpp_result_gen = Rcpp::wrap(jointTauStarFromDef(X, Y, xOnOffVec, yOnOffVec));
    return rcpp_result_gen;
END_RCPP
}
// jointTauStarApprox
double jointTauStarApprox(const arma::mat& X, const arma::mat& Y, const arma::uvec& xOnOffVec, const arma::uvec& yOnOffVec, int sims, int seconds);
RcppExport SEXP SymRC_jointTauStarApprox(SEXP XSEXP, SEXP YSEXP, SEXP xOnOffVecSEXP, SEXP yOnOffVecSEXP, SEXP simsSEXP, SEXP secondsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xOnOffVec(xOnOffVecSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yOnOffVec(yOnOffVecSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    Rcpp::traits::input_parameter< int >::type seconds(secondsSEXP);
    rcpp_result_gen = Rcpp::wrap(jointTauStarApprox(X, Y, xOnOffVec, yOnOffVec, sims, seconds));
    return rcpp_result_gen;
END_RCPP
}
// fullLexTauStarFromDef
double fullLexTauStarFromDef(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_fullLexTauStarFromDef(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(fullLexTauStarFromDef(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// fullLexTauStarApprox
double fullLexTauStarApprox(const arma::mat& X, const arma::mat& Y, int sims);
RcppExport SEXP SymRC_fullLexTauStarApprox(SEXP XSEXP, SEXP YSEXP, SEXP simsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    rcpp_result_gen = Rcpp::wrap(fullLexTauStarApprox(X, Y, sims));
    return rcpp_result_gen;
END_RCPP
}
// lexTauStarFromDef
double lexTauStarFromDef(const arma::mat& X, const arma::mat& Y, const arma::uvec& xPerm, const arma::uvec& yPerm);
RcppExport SEXP SymRC_lexTauStarFromDef(SEXP XSEXP, SEXP YSEXP, SEXP xPermSEXP, SEXP yPermSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xPerm(xPermSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yPerm(yPermSEXP);
    rcpp_result_gen = Rcpp::wrap(lexTauStarFromDef(X, Y, xPerm, yPerm));
    return rcpp_result_gen;
END_RCPP
}
// lexTauStarApprox
double lexTauStarApprox(const arma::mat& X, const arma::mat& Y, const arma::uvec& xPerm, const arma::uvec& yPerm, int sims);
RcppExport SEXP SymRC_lexTauStarApprox(SEXP XSEXP, SEXP YSEXP, SEXP xPermSEXP, SEXP yPermSEXP, SEXP simsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xPerm(xPermSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yPerm(yPermSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    rcpp_result_gen = Rcpp::wrap(lexTauStarApprox(X, Y, xPerm, yPerm, sims));
    return rcpp_result_gen;
END_RCPP
}
// partialTauStarRangeTree
double partialTauStarRangeTree(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_partialTauStarRangeTree(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(partialTauStarRangeTree(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// partialTauStarNaive
double partialTauStarNaive(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_partialTauStarNaive(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(partialTauStarNaive(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// partialTauStarFromDef
double partialTauStarFromDef(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_partialTauStarFromDef(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(partialTauStarFromDef(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// partialTauStarApprox
double partialTauStarApprox(const arma::mat& X, const arma::mat& Y, int sims, int seconds);
RcppExport SEXP SymRC_partialTauStarApprox(SEXP XSEXP, SEXP YSEXP, SEXP simsSEXP, SEXP secondsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    Rcpp::traits::input_parameter< int >::type seconds(secondsSEXP);
    rcpp_result_gen = Rcpp::wrap(partialTauStarApprox(X, Y, sims, seconds));
    return rcpp_result_gen;
END_RCPP
}
// ismRangeTree
double ismRangeTree(const arma::mat& X, const arma::mat& Y, const arma::uvec& xInds0, const arma::uvec& xInds1, const arma::uvec& yInds0, const arma::uvec& yInds1);
RcppExport SEXP SymRC_ismRangeTree(SEXP XSEXP, SEXP YSEXP, SEXP xInds0SEXP, SEXP xInds1SEXP, SEXP yInds0SEXP, SEXP yInds1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds0(xInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds1(xInds1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds0(yInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds1(yInds1SEXP);
    rcpp_result_gen = Rcpp::wrap(ismRangeTree(X, Y, xInds0, xInds1, yInds0, yInds1));
    return rcpp_result_gen;
END_RCPP
}
// ismFromDef
double ismFromDef(const arma::mat& X, const arma::mat& Y, const arma::uvec& xInds0, const arma::uvec& xInds1, const arma::uvec& yInds0, const arma::uvec& yInds1);
RcppExport SEXP SymRC_ismFromDef(SEXP XSEXP, SEXP YSEXP, SEXP xInds0SEXP, SEXP xInds1SEXP, SEXP yInds0SEXP, SEXP yInds1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds0(xInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds1(xInds1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds0(yInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds1(yInds1SEXP);
    rcpp_result_gen = Rcpp::wrap(ismFromDef(X, Y, xInds0, xInds1, yInds0, yInds1));
    return rcpp_result_gen;
END_RCPP
}
// ismApprox
double ismApprox(const arma::mat& X, const arma::mat& Y, const arma::uvec& xInds0, const arma::uvec& xInds1, const arma::uvec& yInds0, const arma::uvec& yInds1, int sims);
RcppExport SEXP SymRC_ismApprox(SEXP XSEXP, SEXP YSEXP, SEXP xInds0SEXP, SEXP xInds1SEXP, SEXP yInds0SEXP, SEXP yInds1SEXP, SEXP simsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds0(xInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type xInds1(xInds1SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds0(yInds0SEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type yInds1(yInds1SEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    rcpp_result_gen = Rcpp::wrap(ismApprox(X, Y, xInds0, xInds1, yInds0, yInds1, sims));
    return rcpp_result_gen;
END_RCPP
}
// kendallsTauNaive
double kendallsTauNaive(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_kendallsTauNaive(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(kendallsTauNaive(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// kendallsTauApprox
double kendallsTauApprox(const arma::mat& X, const arma::mat& Y, int sims);
RcppExport SEXP SymRC_kendallsTauApprox(SEXP XSEXP, SEXP YSEXP, SEXP simsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    rcpp_result_gen = Rcpp::wrap(kendallsTauApprox(X, Y, sims));
    return rcpp_result_gen;
END_RCPP
}
// spearmansRhoNaive
double spearmansRhoNaive(const arma::mat& X, const arma::mat& Y);
RcppExport SEXP SymRC_spearmansRhoNaive(SEXP XSEXP, SEXP YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    rcpp_result_gen = Rcpp::wrap(spearmansRhoNaive(X, Y));
    return rcpp_result_gen;
END_RCPP
}
// spearmansRhoApprox
double spearmansRhoApprox(const arma::mat& X, const arma::mat& Y, int sims);
RcppExport SEXP SymRC_spearmansRhoApprox(SEXP XSEXP, SEXP YSEXP, SEXP simsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type sims(simsSEXP);
    rcpp_result_gen = Rcpp::wrap(spearmansRhoApprox(X, Y, sims));
    return rcpp_result_gen;
END_RCPP
}
// orthRangeTensorCount
arma::uvec orthRangeTensorCount(const arma::mat& samples, const arma::umat& lowerMat, const arma::umat& upperMat);
RcppExport SEXP SymRC_orthRangeTensorCount(SEXP samplesSEXP, SEXP lowerMatSEXP, SEXP upperMatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type lowerMat(lowerMatSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type upperMat(upperMatSEXP);
    rcpp_result_gen = Rcpp::wrap(orthRangeTensorCount(samples, lowerMat, upperMat));
    return rcpp_result_gen;
END_RCPP
}
// alignedRangeTreeCount
arma::uvec alignedRangeTreeCount(const arma::mat& samples, const arma::umat& lowerMat, const arma::umat& upperMat);
RcppExport SEXP SymRC_alignedRangeTreeCount(SEXP samplesSEXP, SEXP lowerMatSEXP, SEXP upperMatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type lowerMat(lowerMatSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type upperMat(upperMatSEXP);
    rcpp_result_gen = Rcpp::wrap(alignedRangeTreeCount(samples, lowerMat, upperMat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"SymRC_powerSetMat", (DL_FUNC) &SymRC_powerSetMat, 1},
    {"SymRC_toJointRankMatrix", (DL_FUNC) &SymRC_toJointRankMatrix, 1},
    {"SymRC_intersectSorted", (DL_FUNC) &SymRC_intersectSorted, 2},
    {"SymRC_unionSorted", (DL_FUNC) &SymRC_unionSorted, 2},
    {"SymRC_setDiffSorted", (DL_FUNC) &SymRC_setDiffSorted, 2},
    {"SymRC_complementSorted", (DL_FUNC) &SymRC_complementSorted, 2},
    {"SymRC_intToUVec", (DL_FUNC) &SymRC_intToUVec, 2},
    {"SymRC_zeroOneVecToInt", (DL_FUNC) &SymRC_zeroOneVecToInt, 1},
    {"SymRC_intPow", (DL_FUNC) &SymRC_intPow, 2},
    {"SymRC_permutations", (DL_FUNC) &SymRC_permutations, 1},
    {"SymRC_orderStats", (DL_FUNC) &SymRC_orderStats, 1},
    {"SymRC_hoeffdingROrthTensor", (DL_FUNC) &SymRC_hoeffdingROrthTensor, 2},
    {"SymRC_hoeffdingRNaive", (DL_FUNC) &SymRC_hoeffdingRNaive, 2},
    {"SymRC_hoeffdingRFromDef", (DL_FUNC) &SymRC_hoeffdingRFromDef, 2},
    {"SymRC_hoeffdingDRangeTree", (DL_FUNC) &SymRC_hoeffdingDRangeTree, 2},
    {"SymRC_hoeffdingDNaive", (DL_FUNC) &SymRC_hoeffdingDNaive, 2},
    {"SymRC_hoeffdingDFromDef", (DL_FUNC) &SymRC_hoeffdingDFromDef, 2},
    {"SymRC_jointTauStarRangeTree", (DL_FUNC) &SymRC_jointTauStarRangeTree, 4},
    {"SymRC_jointTauStarNaive", (DL_FUNC) &SymRC_jointTauStarNaive, 4},
    {"SymRC_jointTauStarFromDef", (DL_FUNC) &SymRC_jointTauStarFromDef, 4},
    {"SymRC_jointTauStarApprox", (DL_FUNC) &SymRC_jointTauStarApprox, 6},
    {"SymRC_fullLexTauStarFromDef", (DL_FUNC) &SymRC_fullLexTauStarFromDef, 2},
    {"SymRC_fullLexTauStarApprox", (DL_FUNC) &SymRC_fullLexTauStarApprox, 3},
    {"SymRC_lexTauStarFromDef", (DL_FUNC) &SymRC_lexTauStarFromDef, 4},
    {"SymRC_lexTauStarApprox", (DL_FUNC) &SymRC_lexTauStarApprox, 5},
    {"SymRC_partialTauStarRangeTree", (DL_FUNC) &SymRC_partialTauStarRangeTree, 2},
    {"SymRC_partialTauStarNaive", (DL_FUNC) &SymRC_partialTauStarNaive, 2},
    {"SymRC_partialTauStarFromDef", (DL_FUNC) &SymRC_partialTauStarFromDef, 2},
    {"SymRC_partialTauStarApprox", (DL_FUNC) &SymRC_partialTauStarApprox, 4},
    {"SymRC_ismRangeTree", (DL_FUNC) &SymRC_ismRangeTree, 6},
    {"SymRC_ismFromDef", (DL_FUNC) &SymRC_ismFromDef, 6},
    {"SymRC_ismApprox", (DL_FUNC) &SymRC_ismApprox, 7},
    {"SymRC_kendallsTauNaive", (DL_FUNC) &SymRC_kendallsTauNaive, 2},
    {"SymRC_kendallsTauApprox", (DL_FUNC) &SymRC_kendallsTauApprox, 3},
    {"SymRC_spearmansRhoNaive", (DL_FUNC) &SymRC_spearmansRhoNaive, 2},
    {"SymRC_spearmansRhoApprox", (DL_FUNC) &SymRC_spearmansRhoApprox, 3},
    {"SymRC_orthRangeTensorCount", (DL_FUNC) &SymRC_orthRangeTensorCount, 3},
    {"SymRC_alignedRangeTreeCount", (DL_FUNC) &SymRC_alignedRangeTreeCount, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_SymRC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
