#ifndef FINAL_MINMAXKMEANS_H
#define FINAL_MINMAXKMEANS_H

#include "structs.h"

void getEuclideanDist(const float **matrix, float **centroids, float **euclideanDistances, mtxDimensions job);

int getNearestCluster(const float *distances, int numClusters);

void cleanOldKMeans(kMeansOutput oldKMeans);

kMeansOutput getKMeansResult(
        const float **matrix,
        float **centroids,
        mtxDimensions jobInfo,
        int getRandom);

#endif //FINAL_MINMAXKMEANS_H
