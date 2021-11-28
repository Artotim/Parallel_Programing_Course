#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "updateWeights.h"
#include "array_routines.h"

#define BLOCK_SZ 256

int *getArrayOfOnes(const long numObjs) {
    /* Initialize an array of ones */

    int *allOneArray = intArrayAlloc(numObjs, 0);
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        allOneArray[objIdx] = 1;
    }

    return allOneArray;
}


void cpuGetScales(const float *matrix, const mtxDimensions job, const int *membership, float *scales) {
    /* Get scales on CPU */


    for (int objIdx = job.d_numObjs; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
        #pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            scales[(clusterIdx * job.numColumns) + columnIdx] += matrix[(objIdx * job.numColumns) + columnIdx];
        }
    }
}

__global__
void d_getScales(const float *d_matrix, const mtxDimensions job, const int *d_membership, float *d_scales) {
    /* Loop over objects to get sum of points */

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (columnIdx < job.numColumns) {
        for (int objIdx = 0; objIdx < job.d_numObjs; objIdx++) {
            int clusterIdx = d_membership[objIdx];
            d_scales[(clusterIdx * job.numColumns) + columnIdx] += d_matrix[(objIdx * job.numColumns) + columnIdx];
        }
    }
}


void resolveScales(const float *matrix, const float *d_matrix, const mtxDimensions job, const int *membership,
                   const int *d_membership, float *d_scales) {
    /* Resolves scales in utilizing CUDA and OpenMP if necessary */

    const long numClusters = job.numClusters, numColumns = job.numColumns /

    d_getScales<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> > (d_matrix, job, d_membership, d_scales);

    if (!job.fit) {
        float *scales = floatArrayAlloc(numClusters * numColumns, 0);
        cudaMemcpy(scales, d_scales, sizeof(float) * numClusters * numColumns, cudaMemcpyDefault);
        cpuGetScales(matrix, job, membership, scales); // rodar após
        cudaMemcpy(d_scales, scales, sizeof(float) * numClusters * numColumns, cudaMemcpyHostToDevice);
        free(scales);
    }
}


__global__
void d_getClusterCount(const int *d_membership, int *d_clusterCount, const mtxDimensions job) {
    /* Get cluster size sequentially */

    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        if (d_membership[objIdx] == clusterIdx) {
            d_clusterCount[clusterIdx]++;
        }
    }
}


__global__
void d_getScalesMean(float *d_scales, int const *d_clusterCount, const long numColumns) {
    /* Loop over clusters to get mean of points */

    int columnIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;

    d_scales[(clusterIdx * numColumns) + columnIdx] =
            d_scales[(clusterIdx * numColumns) + columnIdx] / (float) d_clusterCount[clusterIdx];
}


void cpuGetWeightedDistances(const float *matrix, const float *scales, float *distToCenter, const int *membership,
                             const mtxDimensions job) {
    /* Loop over objects to get sum of distances */

    const long numColumns = job.numColumns;

    for (int objIdx = job.d_numObjs; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
        #pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            distToCenter[(clusterIdx * numColumns) + columnIdx] += powf(
                    matrix[(objIdx * numColumns) + columnIdx] - scales[(clusterIdx * numColumns) + columnIdx], 2);
        }
    }
}


__global__
void d_getWeightedDistances(const float *d_matrix, const float *d_scales, float *d_distToCenter,
                            const int *d_membership, const mtxDimensions job) {
    /* Loop over clusters to get mean of points */

    int numColumns = job.numColumns;
    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (columnIdx < numColumns) {
        for (int objIdx = 0; objIdx < job.d_numObjs; objIdx++) {
            int clusterIdx = d_membership[objIdx];
            d_distToCenter[(clusterIdx * numColumns) + columnIdx] +=
                    powf(d_matrix[(objIdx * numColumns) + columnIdx] - d_scales[(clusterIdx * numColumns) + columnIdx],
                         2);
        }
    }
}


void resolveWeightedDistances(const float *matrix, const float *d_matrix, const float *d_scales, float *d_distToCenter,
                              const int *membership, const int *d_membership, const mtxDimensions job) {
    /* Resolve weighted distances in CUDA and OpenMP if necessary */

    const long numColumns = job.numColumns, numClusters = job.numClusters;

    d_getWeightedDistances <<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> >
                                                             (d_matrix, d_scales, d_distToCenter, d_membership, job);

    if (!job.fit) {
        float *scales = floatArrayAlloc(numClusters * numColumns, 0);
        cudaMemcpy(scales, d_scales, sizeof(float) * numClusters * numColumns, cudaMemcpyDefault);
        float *distToCenter = floatArrayAlloc(numClusters * numColumns, 0);
        cudaMemcpy(distToCenter, d_distToCenter, sizeof(float) * numClusters * numColumns, cudaMemcpyDefault);
        cpuGetWeightedDistances(matrix, scales, distToCenter, membership, job); // rodar após
        cudaMemcpy(d_distToCenter, distToCenter, sizeof(float) * numClusters * numColumns, cudaMemcpyHostToDevice);

        free(scales);
        free(distToCenter);
    }
}


__global__
void d_solveWcss(float *d_distToCenter, int *d_clusterCount, const float *d_clusterWeights,
                 float *d_wcss, const float pValue, const mtxDimensions job) {
    /* Return array with weighted sum of distances */

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (columnIdx >= job.numColumns) return;

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        if (d_clusterCount[clusterIdx] > 1) {
            if (d_clusterWeights) {
                d_wcss[columnIdx] +=
                        d_distToCenter[(clusterIdx * job.numColumns) + columnIdx] *
                        powf(d_clusterWeights[clusterIdx], pValue);
            } else {
                d_wcss[columnIdx] += d_distToCenter[(clusterIdx * job.numColumns) + columnIdx];
            }
        }
    }

}


float *getWcss(const float *matrix, const float *d_matrix, const mtxDimensions job, const int *membership,
               const float *d_clusterWeights, float pValue) {
    /* Get sum of distances of every column to calculate weights*/

    const long numColumns = job.numColumns, numClusters = job.numClusters;
    int *d_membership = cudaIntAlloc(job.numObjs, 0);
    cudaMemcpy(d_membership, membership, sizeof(float) * job.numObjs, cudaMemcpyDefault);

    float *distToCenter = floatArrayAlloc(numClusters * numColumns, 1);
    int *clusterCount = intArrayAlloc(job.numClusters, 1);

    float *d_scales = cudaFloatAlloc(numClusters * numColumns, 1);
    float *d_distToCenter = cudaFloatAlloc(numClusters * numColumns, 1);
    int *d_clusterCount = cudaIntAlloc(numClusters, 1);

    resolveScales(matrix, d_matrix, job, membership, d_membership, d_scales);
    d_getClusterCount<<<1, numClusters >> > (d_membership, d_clusterCount, job);


    d_getScalesMean<<<numColumns, numClusters >> > (d_scales, d_clusterCount, job.numColumns);


    resolveWeightedDistances(matrix, d_matrix, d_scales, d_distToCenter, membership, d_membership, job);


    // Loop over objects to get sum of distances
    float *d_wcss = cudaFloatAlloc(numColumns, 1);
    d_solveWcss<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> >
                                                 (d_distToCenter, d_clusterCount, d_clusterWeights, d_wcss, pValue, job);


    cudaFree(d_scales);
    cudaFree(d_distToCenter);
    cudaFree(d_clusterCount);
    return d_wcss;
}


__global__
void d_subtractLists(float *result, const float *list, const float *subtrahend, long numColumns) {
    /* Subtract two lists of same size value by value */

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (columnIdx < numColumns) {
        result[columnIdx] = list[columnIdx] - subtrahend[columnIdx];
    }
}


float listToNumber(float *vector, const long numColumns) {
    /* Reduce list to number: sqrtf(sum(list^2)) */

    float result = 0;

#pragma omp parallel for reduction(+:result)
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result += powf(vector[columnIdx], 2);
    }

    return sqrtf(result);
}


float *divideLists(const float *list, float divisor, long numColumns) {
    /* Divide two lists of same size value by value */

    float *result = floatArrayAlloc(numColumns, 0);

#pragma omp parallel for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result[columnIdx] = list[columnIdx] / divisor;
    }

    return result;
}


float absoluteSum(const float *list, long numColumns) {
    /* Return sum of absolute values from list */

    float result = 0;
#pragma omp parallel for reduction(+:result)
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (list[columnIdx] > 0) {
            result += list[columnIdx];
        } else {
            result += (list[columnIdx] * -1);
        }
    }
    return result;
}


float maxAbsolute(float *vector, long numColumns) {
    /* Max absolute number in list */

    float ans = FLT_MIN;
#pragma omp parallel for
    for (int j = 0; j < numColumns; j++) {
        if (fabsf(vector[j]) > ans) {
            ans = fabsf(vector[j]);
        }
    }
    return ans;
}


float *soft(float *vector, float lam, long numColumns) {
    /* Return list of values minus lam if greater than 0 */

    float *softened = floatArrayAlloc(numColumns, 0);

#pragma omp parallel for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {

        float value = fabsf(vector[columnIdx]) - lam;
        if (value > 0) {
            softened[columnIdx] = vector[columnIdx] > 0 ? value : (float) -1 * value;
        } else {
            softened[columnIdx] = 0;
        }
    }

    return softened;
}


float calculateLam(float *array, float lasso, const long numColumns) {
    /* Calculates lam for array*/

    float listNumber = listToNumber(array, numColumns);
    float *dividedList = divideLists(array, listNumber, numColumns);
    float *dividedSoftened;


    if (listNumber == 0 || absoluteSum(dividedList, numColumns) <= lasso) {
        free(dividedList);
        return 0;
    }


    float lam1 = 0;
    float lam2 = maxAbsolute(array, numColumns) - 1e-05f;
    float *softened, softenedNumber;

    int iter = 1;
    // Loops until lam2 - lam1 convert to 0
    while (iter <= 15 && (lam2 - lam1) > 1e-04) {
        softened = soft(array, (lam1 + lam2) / 2, numColumns);

        softenedNumber = listToNumber(softened, numColumns);

        dividedSoftened = divideLists(softened, softenedNumber, numColumns);

        if (absoluteSum(dividedSoftened, numColumns) < lasso) {
            lam2 = (lam1 + lam2) / 2;
        } else {
            lam1 = (lam1 + lam2) / 2;
        }

        free(dividedSoftened);
        free(softened);

        iter++;
    }

    free(dividedList);

    return (lam1 + lam2) / 2;
}


void updatesWeights(const float *matrix, const float *d_matrix, const mtxDimensions jobInfo, int *membership,
                    float l1Bound, float *clusterWeights, float pValue, float *weights) {
    /* Main routine to update column weights */
    printf("Performing weights update...");

    float *d_clusterWeights = cudaFloatAlloc(jobInfo.numClusters, 0);
    cudaMemcpy(d_clusterWeights, clusterWeights, sizeof(float) * jobInfo.numClusters, cudaMemcpyDefault);

    const long numColumns = jobInfo.numColumns;

    float *d_wcssPerFeature = getWcss(matrix, d_matrix, jobInfo, membership, d_clusterWeights, pValue);

    int *arrayOfOnes = getArrayOfOnes(jobInfo.numObjs);
    float *d_tssPerFeature = getWcss(matrix, d_matrix, jobInfo, arrayOfOnes, NULL, pValue);
    free(arrayOfOnes);


    float *d_weightsDifference = cudaFloatAlloc(numColumns, 0);
    d_subtractLists<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> >
                                                     (d_weightsDifference, d_tssPerFeature, d_wcssPerFeature, numColumns);

    float *weightsDifference = floatArrayAlloc(numColumns, 0);
    cudaMemcpy(weightsDifference, d_weightsDifference, sizeof(float) * numColumns, cudaMemcpyDefault);

    float lam = calculateLam(weightsDifference, l1Bound, numColumns);

    float *weightsUnscaled = soft(weightsDifference, lam, numColumns);

    float *resultWeights = divideLists(weightsUnscaled, listToNumber(weightsUnscaled, numColumns), numColumns);
    memcpy(weights, resultWeights, sizeof(float) * numColumns);

    cudaFree(d_wcssPerFeature);
    cudaFree(d_tssPerFeature);
    free(weightsDifference);
    free(weightsUnscaled);
    free(resultWeights);

    printf("Done\n");
}
