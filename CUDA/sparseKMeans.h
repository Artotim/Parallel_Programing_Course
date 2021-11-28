#ifndef FINAL_PARALLEL_SPARSEKMEANS_H
#define FINAL_PARALLEL_SPARSEKMEANS_H

#include "structs.h"

int getNonZeroWeights(const float *columnWeights, long numColumns);

kMeansOutput sparseKMeans(const float *matrix, const mtxDimensions jobInfo, float *columnsWeights, const int maxLoops);

#endif //FINAL_PARALLEL_SPARSEKMEANS_H
