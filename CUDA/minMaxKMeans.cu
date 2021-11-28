#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#include "structs.h"
#include "minMaxKMeans.h"
#include "array_routines.h"

#define BLOCK_SZ 256

void initializeRandomCenters(const float *matrix, float *centroids, const mtxDimensions job) {
    /* Initialize centroids to random points */

    int randomPoint;

    srand(time(NULL) * job.numObjs);

    printf("\nSetting random centers to");

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        randomPoint = rand() % job.numObjs;
        printf(" %d", randomPoint);

#pragma omp parallel for firstprivate(randomPoint, clusterIdx)
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[clusterIdx * job.numColumns + columnIdx] = matrix[randomPoint * job.numColumns +
                                                                        columnIdx];
        }
    }

    printf(".\n\n");
}


__global__
void d_initializeClusterWeights(float *d_clusterWeights, float *d_oldClusterWeights, const long numClusters) {
    /* Intialize cluster weights */

    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (clusterIdx < numClusters) {
        d_clusterWeights[clusterIdx] = 1 / (float) numClusters;
        d_oldClusterWeights[clusterIdx] = 1 / (float) numClusters;
    }
}


__global__
void d_getEuclideanDist(const float *d_matrix, float *d_centroids, float *d_euclideanDistances,
                        const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters on GPU*/

    const long numColumns = job.numColumns;
    int objIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;
    float distance = 0;
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        distance += powf(
                (d_matrix[(objIdx * numColumns) + columnIdx] - d_centroids[(clusterIdx * numColumns) + columnIdx]), 2);
    }

    d_euclideanDistances[(objIdx * job.numClusters) + clusterIdx] = sqrtf(distance);
}


void cpuGetEuclideanDist(const float *matrix, float *centroids, float *euclideanDistances, const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters on cpu */

    const long numColumns = job.numColumns;


    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
#pragma omp parallel for
        for (int objIdx = job.d_numObjs; objIdx < job.numObjs; objIdx++) {
            float distance = 0;
            for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
                distance += powf((matrix[(objIdx * numColumns) + columnIdx] -
                                  centroids[(clusterIdx * numColumns) + columnIdx]),
                                 2);
            }

            euclideanDistances[((objIdx - job.d_numObjs) * job.numClusters) + clusterIdx] = sqrtf(distance);
        }
    }

}


void resolveEuclideanDistances(const float *matrix, const float *d_matrix, float *centroids, float *d_centroids,
                               float *d_euclideanDistances, const mtxDimensions job) {
    /* Resolve euclidean distances using CUDA and OpenMP if necessary */

    d_getEuclideanDist<<<job.d_numObjs, job.numClusters >> > (d_matrix, d_centroids, d_euclideanDistances, job);

    if (!job.fit) {
        cudaMemcpy(centroids, d_centroids, sizeof(float) * job.numClusters * job.numColumns, cudaMemcpyDefault);
        float *tempEuclideanDistances = floatArrayAlloc(job.cpuNumObjs * job.numClusters, 1);
        cpuGetEuclideanDist(matrix, centroids, tempEuclideanDistances, job); // pode rodar junto com o cuda
        cudaMemcpy(d_euclideanDistances + job.d_numObjs * job.numClusters, tempEuclideanDistances,
                   sizeof(float) * job.cpuNumObjs * job.numClusters, cudaMemcpyDefault);
        free(tempEuclideanDistances);

    }

}


__global__
void d_initializeDistanceWeights(float *d_weightedDistances, const float *d_clusterWeights,
                                 const float *d_euclideanDistances, const mtxDimensions job, const float pValue) {
    /* Initialize weighted distances */

    int objIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;


    d_weightedDistances[objIdx + clusterIdx * job.numObjs] =
            powf(d_clusterWeights[clusterIdx], pValue) * d_euclideanDistances[objIdx + clusterIdx * job.numObjs];

}


__global__
void d_findNearestCluster(const float *d_distances, float *d_objsMtxMembership, int *d_objsMembership,
                          const mtxDimensions job) {
    /* Find nearest cluster for object */

    int objIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (objIdx >= job.numObjs) return;

    float min_dist = FLT_MAX;
    int nearestCluster = 0;

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        if (min_dist > d_distances[objIdx * job.numClusters + clusterIdx]) {
            min_dist = d_distances[objIdx * job.numClusters + clusterIdx];
            nearestCluster = clusterIdx;
        }
    }

    d_objsMtxMembership[(objIdx * job.numClusters) + nearestCluster] = 1;
    d_objsMembership[objIdx] = nearestCluster;

}


__global__
void d_getClusterSize(const int *d_objsMembership, int *d_clusterSize, const mtxDimensions job) {
    /* Get cluster size sequentially */

    for (int cluster = 0; cluster < job.numClusters; cluster++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            if (d_objsMembership[objIdx] == cluster)
                d_clusterSize[cluster]++;
        }
    }
}


int *getObjectsDistribution(int *d_clusterSize, const float *d_weightedDistances, float *d_objsMtxMembership,
                            int *d_objsMembership, const mtxDimensions job) {
    /* Gets nearest cluster for each object, cluster sizes and clusters membership array */

    const long numObjs = job.numObjs, numClusters = job.numClusters;
    int *clusterSize = intArrayAlloc(numClusters, 1);


    d_findNearestCluster<<<(numObjs + 255) / BLOCK_SZ, BLOCK_SZ >> >
                                                       (d_weightedDistances, d_objsMtxMembership, d_objsMembership, job);


    d_getClusterSize<<<1, 1 >> > (d_objsMembership, d_clusterSize, job);
    cudaMemcpy(clusterSize, d_clusterSize, sizeof(int) * numClusters, cudaMemcpyDefault);


    return clusterSize;
}


void cpuClusterMeans(const float *matrix, float *centroids, const int *objsMembership, const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    const long numColumns = job.numColumns;


    for (int objIdx = job.d_numObjs; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = objsMembership[objIdx];

#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            centroids[(clusterIdx * numColumns) + columnIdx] += matrix[(objIdx * numColumns) + columnIdx];
        }
    }
}


__global__
void d_getClusterMeans(const float *d_matrix, float *d_centroids, const int *d_objsMembership,
                       const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    int columnIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;


    int cluster;
    d_centroids[(clusterIdx * job.numColumns) + columnIdx] = 0;

    // Loop over matrix to populate sumDistances
    for (int objIdx = 0; objIdx < job.d_numObjs; objIdx++) {
        cluster = d_objsMembership[objIdx];
        if (cluster == clusterIdx) {
            d_centroids[(clusterIdx * job.numColumns) + columnIdx] += d_matrix[(objIdx * job.numColumns) + columnIdx];
        }
    }

}


__global__
void d_setMeans(float *d_centroids, const int *d_clusterSize, const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    int columnIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;


    // Loop over sumDistances to get means and move centroids
    d_centroids[(clusterIdx * job.numColumns) + columnIdx] /= (float) d_clusterSize[clusterIdx];
}


void resolveClusterMeans(const float *matrix, const float *d_matrix, float *centroids, float *d_centroids,
                         const int *d_clusterSize, int *objsMembership, int *d_objsMembership,
                         const mtxDimensions job) {
    /* Resolve cluster means with CUDA and OpenMP if necessary */

    const long numClusters = job.numClusters, numColumns = job.numColumns;


    d_getClusterMeans<<< job.numColumns, job.numClusters >> > (d_matrix, d_centroids, d_objsMembership, job);

    if (!job.fit) {
        cudaMemcpy(objsMembership, d_objsMembership, sizeof(int) * job.numObjs, cudaMemcpyDefault);
        cudaMemcpy(centroids, d_centroids, sizeof(float) * numClusters * numColumns, cudaMemcpyDefault);
        cpuClusterMeans(matrix, centroids, objsMembership, job); // rodar após
        cudaMemcpy(d_centroids, centroids, sizeof(float) * numClusters * numColumns, cudaMemcpyHostToDevice);
    }

    d_setMeans<<< job.numColumns, job.numClusters >> > (d_centroids, d_clusterSize, job);
}


__global__
void d_updateClusterWeights(float *d_clusterWeights, float *d_objsMtxMembership, float *d_euclideanDistances,
                            const mtxDimensions job, const float pValue) {
    /* Updates cluster weights according to distances and size */

    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    const long numClusters = job.numClusters;
    __shared__ float sumWeightedDistances;
    sumWeightedDistances = 0;
    __syncthreads();


    float clustersDistance = 0;
    // Loop over clusters to get sum of distances for each obj
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        clustersDistance +=
                d_objsMtxMembership[(objIdx * numClusters) + clusterIdx] *
                d_euclideanDistances[(objIdx * numClusters) + clusterIdx];

    }
    atomicAdd(&sumWeightedDistances, powf(clustersDistance, (1 / (1 - pValue))));
    __syncthreads();

    d_clusterWeights[clusterIdx] = powf(clustersDistance, (1 / (1 - pValue))) / sumWeightedDistances;
}


float updateSumWeights(float *weightedDistances, const float *euclideanDistances, float *clusterWeights,
                       const float pValue, const mtxDimensions job) {
    /* Get sum of weights to check condition */

    const long numClusters = job.numClusters;
    float sumWeights = 0;

#pragma omp parallel for reduction(+:sumWeights)
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx * numClusters + clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) *
                    euclideanDistances[(objIdx * numClusters) + clusterIdx];
            sumWeights += weightedDistances[objIdx * numClusters + clusterIdx];
        }
    }

    return sumWeights;
}


kMeansOutput minMaxKMeans(const float *matrix, const float *d_matrix, float *centroids, const mtxDimensions jobInfo,
                          const int getRandomPoints) {
    /* Main routine for MinMaxKMeans */

    // Initialize variables
    const long numObjs = jobInfo.numObjs, numClusters = jobInfo.numClusters;
    kMeansOutput kMeansResult;
    float pValue = 0;
    const float pStep = 0.01f, pMax = 0.5f;
    int flag = 1;

    // If random is true, initialize centers with random points
    if (getRandomPoints == 1) {
        initializeRandomCenters(matrix, centroids, jobInfo);
    }

    // DELETAR PAR A RANDOMIZAR
    if (jobInfo.numClusters == 2) {
        for (int j = 0; j < jobInfo.numColumns; j++) {
            centroids[0 + j] = matrix[0 + j];
            centroids[1 * jobInfo.numColumns + j] = matrix[2 * jobInfo.numColumns + j];
        }
    }
    float *d_centroids = cudaFloatAlloc(numClusters * jobInfo.numColumns, 0);
    cudaMemcpy(d_centroids, centroids, sizeof(float) * numClusters * jobInfo.numColumns, cudaMemcpyDefault);

    printf("Performing MinMaxKMeans...");

    // Initialize matrices and arrays to be used
    float *oldObjsMembership = floatArrayAlloc(numObjs * numClusters, 1);
    float *objsMtxMembership = floatArrayAlloc(numObjs * numClusters, 1);

    int *objsMembership = intArrayAlloc(numObjs, 0);


    // Initialize cluster weights
    float *clusterWeights = floatArrayAlloc(numClusters, 0);
    float *d_clusterWeights = cudaFloatAlloc(numClusters, 0);
    float *d_oldClusterWeights = cudaFloatAlloc(numClusters, 0);
    d_initializeClusterWeights<<<1, numClusters >> > (d_clusterWeights, d_oldClusterWeights, numClusters);



    // Initialize euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 1);
    float *d_euclideanDistances = cudaFloatAlloc(numObjs * numClusters, 0);
    resolveEuclideanDistances(matrix, d_matrix, centroids, d_centroids, d_euclideanDistances, jobInfo);


    // Initialize matrix weights
    float *weightedDistances = floatArrayAlloc(numObjs * numClusters, 0);
    float *d_weightedDistances = cudaFloatAlloc(numObjs * numClusters, 0);
    d_initializeDistanceWeights <<< numObjs, numClusters >> >
                                             (d_weightedDistances, d_clusterWeights, d_euclideanDistances, jobInfo, pValue);


    float *d_oldObjsMembership = cudaFloatAlloc(numObjs * numClusters, 1);
    float *d_objsMtxMembership = cudaFloatAlloc(numObjs * numClusters, 1);
    int *d_objsMembership = cudaIntAlloc(numObjs, 0);



    // Variables for loop
    int empty = 0, nIter = 0;
    float oldSumWeights, sumWeights = 0;


    // Loop until weights converge to 0 or hit max loops
    while (1) {
        nIter++;

        // Classify each object and get cluster sizes
        int *d_clusterSize = cudaIntAlloc(numClusters, 1);
        int *clusterSize = getObjectsDistribution(d_clusterSize, d_weightedDistances, d_objsMtxMembership,
                                                  d_objsMembership, jobInfo);

        // Lower pValue if some cluster is empty
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                cudaMemcpy(d_objsMtxMembership, d_oldObjsMembership, sizeof(float) * numObjs * numClusters,
                           cudaMemcpyDefault);
                cudaMemcpy(d_clusterWeights, d_oldClusterWeights, sizeof(float) * numClusters, cudaMemcpyDefault);
                break;
            }
        }

        // Break loop if pValue lesser than 0
        if (pValue < 0) {
            pValue = 0;
            flag = 0;
            free(clusterSize);
            cudaFree(d_clusterSize);
            break;
        }

        // Update centroids with distances means
        resolveClusterMeans(matrix, d_matrix, centroids, d_centroids, d_clusterSize, objsMembership, d_objsMembership,
                            jobInfo);


        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            cudaMemcpy(d_oldObjsMembership, d_objsMtxMembership, sizeof(float) * numObjs * numClusters,
                       cudaMemcpyDefault);
            cudaMemcpy(d_oldClusterWeights, d_clusterWeights, sizeof(float) * numClusters, cudaMemcpyDefault);

            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        resolveEuclideanDistances(matrix, d_matrix, centroids, d_centroids, d_euclideanDistances, jobInfo);

        // Update clusters weights according to distances
        d_updateClusterWeights<<<1, numClusters >> >
                                    (d_clusterWeights, d_objsMtxMembership, d_euclideanDistances, jobInfo, pValue); //ok

        // Update sum of weights to verify condition
        oldSumWeights = sumWeights;

        cudaMemcpy(euclideanDistances, d_euclideanDistances, sizeof(float) * (numObjs * numClusters),
                   cudaMemcpyDefault);
        cudaMemcpy(clusterWeights, d_clusterWeights, sizeof(float) * numClusters, cudaMemcpyDefault);
        sumWeights = updateSumWeights(weightedDistances, euclideanDistances, clusterWeights, pValue, jobInfo);

        free(clusterSize);
        cudaFree(d_clusterSize);

        // Break loop when weights convert to 0
        if (nIter > 1) {
            if (fabsf(sumWeights - oldSumWeights) <= 1e-04 || nIter >= 10) {
                break;
            }
        }
    }

    printf("Done\n");

    // Set return results
    cudaMemcpy(centroids, d_centroids, sizeof(float) * numClusters * jobInfo.numColumns, cudaMemcpyDefault);
    cudaMemcpy(objsMembership, d_objsMembership, sizeof(int) * numObjs, cudaMemcpyDefault);

    kMeansResult.centers = centroids;  // here
    kMeansResult.flag = flag;
    kMeansResult.cluster = objsMembership;  // here
    kMeansResult.clusterWeights = clusterWeights; // here
    kMeansResult.pValue = pValue;

    cudaFree(d_clusterWeights);
    cudaFree(d_oldClusterWeights);
    cudaFree(d_euclideanDistances);
    cudaFree(d_weightedDistances);
    cudaFree(d_oldObjsMembership);
    cudaFree(d_objsMtxMembership);
    cudaFree(d_objsMembership);

    free(weightedDistances);
    free(objsMtxMembership);
    free(oldObjsMembership);
    free(euclideanDistances);

    return kMeansResult;
}


void cleanOldKMeans(kMeansOutput oldKMeans) {
    /* Free unused kMeans results */

    free(oldKMeans.cluster);
    free(oldKMeans.clusterWeights);
}


kMeansOutput getKMeansResult(const float *matrix, const float *d_matrix, float *centroids, const mtxDimensions jobInfo,
                             int getRandom) {
    /* Perform MinMax KMeans until all clusters are populated */
    int nIter = 0;

    kMeansOutput kMeansResult = minMaxKMeans(matrix, d_matrix, centroids, jobInfo, getRandom);

    while (kMeansResult.flag == 0 && nIter < 6) {
        getRandom = 1;
        nIter++;
        cleanOldKMeans(kMeansResult);
        kMeansResult = minMaxKMeans(matrix, d_matrix, centroids, jobInfo, getRandom);
    }

    return kMeansResult;
}
