
#ifndef FINAL_PARALLEL_SPARSEKMEANS_H
#define FINAL_PARALLEL_SPARSEKMEANS_H

#include "structs.h"

int getNonZeroWeights(const float *columnWeights, long numColumns);

kMeansOutput sparseKMeans(const float *matrix, mtxDimensions jobInfo, float *columnsWeights, int maxLoops);

#endif //FINAL_PARALLEL_SPARSEKMEANS_H
