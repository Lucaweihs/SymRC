# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

powerSetMat <- function(n) {
    .Call('SymRC_powerSetMat', PACKAGE = 'SymRC', n)
}

toJointRankMatrix <- function(samples) {
    .Call('SymRC_toJointRankMatrix', PACKAGE = 'SymRC', samples)
}

intersectSorted <- function(vec1, vec2) {
    .Call('SymRC_intersectSorted', PACKAGE = 'SymRC', vec1, vec2)
}

unionSorted <- function(vec1, vec2) {
    .Call('SymRC_unionSorted', PACKAGE = 'SymRC', vec1, vec2)
}

setDiffSorted <- function(vec1, vec2) {
    .Call('SymRC_setDiffSorted', PACKAGE = 'SymRC', vec1, vec2)
}

complementSorted <- function(inds, maxInd) {
    .Call('SymRC_complementSorted', PACKAGE = 'SymRC', inds, maxInd)
}

intToUVec <- function(uint, length) {
    .Call('SymRC_intToUVec', PACKAGE = 'SymRC', uint, length)
}

zeroOneVecToInt <- function(zeroOneVec) {
    .Call('SymRC_zeroOneVecToInt', PACKAGE = 'SymRC', zeroOneVec)
}

intPow <- function(base, exponent) {
    .Call('SymRC_intPow', PACKAGE = 'SymRC', base, exponent)
}

permutations <- function(n) {
    .Call('SymRC_permutations', PACKAGE = 'SymRC', n)
}

orderStats <- function(M) {
    .Call('SymRC_orderStats', PACKAGE = 'SymRC', M)
}

hoeffdingROrthTensor <- function(X, Y) {
    .Call('SymRC_hoeffdingROrthTensor', PACKAGE = 'SymRC', X, Y)
}

hoeffdingRNaive <- function(X, Y) {
    .Call('SymRC_hoeffdingRNaive', PACKAGE = 'SymRC', X, Y)
}

hoeffdingRFromDef <- function(X, Y) {
    .Call('SymRC_hoeffdingRFromDef', PACKAGE = 'SymRC', X, Y)
}

hoeffdingDRangeTree <- function(X, Y) {
    .Call('SymRC_hoeffdingDRangeTree', PACKAGE = 'SymRC', X, Y)
}

hoeffdingDNaive <- function(X, Y) {
    .Call('SymRC_hoeffdingDNaive', PACKAGE = 'SymRC', X, Y)
}

hoeffdingDFromDef <- function(X, Y) {
    .Call('SymRC_hoeffdingDFromDef', PACKAGE = 'SymRC', X, Y)
}

jointTauStarRangeTree <- function(X, Y, xOnOffVec, yOnOffVec) {
    .Call('SymRC_jointTauStarRangeTree', PACKAGE = 'SymRC', X, Y, xOnOffVec, yOnOffVec)
}

jointTauStarNaive <- function(X, Y, xOnOffVec, yOnOffVec) {
    .Call('SymRC_jointTauStarNaive', PACKAGE = 'SymRC', X, Y, xOnOffVec, yOnOffVec)
}

jointTauStarFromDef <- function(X, Y, xOnOffVec, yOnOffVec) {
    .Call('SymRC_jointTauStarFromDef', PACKAGE = 'SymRC', X, Y, xOnOffVec, yOnOffVec)
}

jointTauStarApprox <- function(X, Y, xOnOffVec, yOnOffVec, sims, seconds) {
    .Call('SymRC_jointTauStarApprox', PACKAGE = 'SymRC', X, Y, xOnOffVec, yOnOffVec, sims, seconds)
}

fullLexTauStarFromDef <- function(X, Y) {
    .Call('SymRC_fullLexTauStarFromDef', PACKAGE = 'SymRC', X, Y)
}

fullLexTauStarApprox <- function(X, Y, sims) {
    .Call('SymRC_fullLexTauStarApprox', PACKAGE = 'SymRC', X, Y, sims)
}

lexTauStarFromDef <- function(X, Y, xPerm, yPerm) {
    .Call('SymRC_lexTauStarFromDef', PACKAGE = 'SymRC', X, Y, xPerm, yPerm)
}

lexTauStarApprox <- function(X, Y, xPerm, yPerm, sims) {
    .Call('SymRC_lexTauStarApprox', PACKAGE = 'SymRC', X, Y, xPerm, yPerm, sims)
}

partialTauStarRangeTree <- function(X, Y) {
    .Call('SymRC_partialTauStarRangeTree', PACKAGE = 'SymRC', X, Y)
}

partialTauStarNaive <- function(X, Y) {
    .Call('SymRC_partialTauStarNaive', PACKAGE = 'SymRC', X, Y)
}

partialTauStarFromDef <- function(X, Y) {
    .Call('SymRC_partialTauStarFromDef', PACKAGE = 'SymRC', X, Y)
}

partialTauStarApprox <- function(X, Y, sims, seconds) {
    .Call('SymRC_partialTauStarApprox', PACKAGE = 'SymRC', X, Y, sims, seconds)
}

ismRangeTree <- function(X, Y, xInds0, xInds1, yInds0, yInds1) {
    .Call('SymRC_ismRangeTree', PACKAGE = 'SymRC', X, Y, xInds0, xInds1, yInds0, yInds1)
}

ismFromDef <- function(X, Y, xInds0, xInds1, yInds0, yInds1) {
    .Call('SymRC_ismFromDef', PACKAGE = 'SymRC', X, Y, xInds0, xInds1, yInds0, yInds1)
}

ismApprox <- function(X, Y, xInds0, xInds1, yInds0, yInds1, sims) {
    .Call('SymRC_ismApprox', PACKAGE = 'SymRC', X, Y, xInds0, xInds1, yInds0, yInds1, sims)
}

kendallsTauNaive <- function(X, Y) {
    .Call('SymRC_kendallsTauNaive', PACKAGE = 'SymRC', X, Y)
}

kendallsTauApprox <- function(X, Y, sims) {
    .Call('SymRC_kendallsTauApprox', PACKAGE = 'SymRC', X, Y, sims)
}

spearmansRhoNaive <- function(X, Y) {
    .Call('SymRC_spearmansRhoNaive', PACKAGE = 'SymRC', X, Y)
}

spearmansRhoApprox <- function(X, Y, sims) {
    .Call('SymRC_spearmansRhoApprox', PACKAGE = 'SymRC', X, Y, sims)
}

orthRangeTensorCount <- function(samples, lowerMat, upperMat) {
    .Call('SymRC_orthRangeTensorCount', PACKAGE = 'SymRC', samples, lowerMat, upperMat)
}

alignedRangeTreeCount <- function(samples, lowerMat, upperMat) {
    .Call('SymRC_alignedRangeTreeCount', PACKAGE = 'SymRC', samples, lowerMat, upperMat)
}

