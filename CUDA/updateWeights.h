#ifndef FINAL_PARALLEL_UPDATEWEIGHTS_H
#define FINAL_PARALLEL_UPDATEWEIGHTS_H
#include "structs.h"

float absoluteSum(const float *list, long numColumns);

void updatesWeights(const float *matrix, const float *d_matrix, mtxDimensions jobInfo, int *membership, float l1Bound,
                    float *clusterWeights, float pValue, float *weights);

#endif //FINAL_PARALLEL_UPDATEWEIGHTS_H
