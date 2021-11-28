#ifndef FINAL_UPDATEWEIGHTS_H
#define FINAL_UPDATEWEIGHTS_H

#include "structs.h"

float absoluteSum(const float *list, long numColumns);

void updatesWeights(const float **matrix, mtxDimensions jobInfo, int *membership, float l1Bound, float *clusterWeights,
                    float pValue, float *weights);

#endif //FINAL_UPDATEWEIGHTS_H
