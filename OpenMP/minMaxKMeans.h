#ifndef FINAL_PARALLEL_MINMAXKMEANS_H
#define FINAL_PARALLEL_MINMAXKMEANS_H
#include "structs.h"

void cpuGetEuclideanDist(const float *matrix, float *centroids, float *euclideanDistances, mtxDimensions job);

void cleanOldKMeans(kMeansOutput oldKMeans);

kMeansOutput getKMeansResult(const float *matrix, float *centroids, mtxDimensions jobInfo, int getRandom);

#endif //FINAL_PARALLEL_MINMAXKMEANS_H
