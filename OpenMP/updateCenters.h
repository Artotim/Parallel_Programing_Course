
#ifndef FINAL_PARALLEL_UPDATECENTERS_H
#define FINAL_PARALLEL_UPDATECENTERS_H
#include "structs.h"

kMeansOutput updateCenters(const float *matrix, mtxDimensions jobInfo, float *columnsWeights,
                           int *membership, int nonZeroWeights);

#endif //FINAL_PARALLEL_UPDATECENTERS_H
