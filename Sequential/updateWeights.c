#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "updateWeights.h"
#include "matrix_array_routines.h"


int *getArrayOfOnes(const long numObjs) {
    /* Initialize an array of ones */

    int *allOneArray = intArrayAlloc(numObjs, 0);
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        allOneArray[objIdx] = 1;
    }

    return allOneArray;
}


float *getScaleSum(float **distToCenter, int *clusterCount, const float *clusterWeights,
                   const float pValue, const mtxDimensions job) {
    /* Return array with weighted sum of distances */

    float *scaleSum = floatArrayAlloc(job.numColumns, 1);

    for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
        for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
            if (clusterCount[clusterIdx] > 1) {
                if (clusterWeights) {
                    scaleSum[columnIdx] +=
                            distToCenter[clusterIdx][columnIdx] * powf(clusterWeights[clusterIdx], pValue);
                } else {
                    scaleSum[columnIdx] += distToCenter[clusterIdx][columnIdx];
                }
            }
        }
    }

    matrixFree(distToCenter, job.numClusters);
    free(clusterCount);

    return scaleSum;
}


float *getWcss(const float **matrix, const mtxDimensions job, const int *membership,
               float *clusterWeights, float pValue) {
    /* Get sum of distances of every column to calculate weights*/

    const long numColumns = job.numColumns, numClusters = job.numClusters;
    int objIdx, columnIdx, clusterIdx;

    float **scales = matrixAlloc(numClusters, numColumns);
    float **distToCenter = matrixAlloc(numClusters, numColumns);
    int *clusterCount = intArrayAlloc(numClusters, 1);


    // Loop over objects to get sum of points
    for (objIdx = 0; objIdx < job.numObjs; objIdx++) {
        clusterIdx = membership[objIdx];
        clusterCount[clusterIdx]++;
        for (columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            scales[clusterIdx][columnIdx] += matrix[objIdx][columnIdx];
        }
    }

    // Loop over clusters to get mean of points
    for (clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            scales[clusterIdx][columnIdx] = scales[clusterIdx][columnIdx] / (float) clusterCount[clusterIdx];
        }
    }

    // Loop over objects to get sum of distances
    for (objIdx = 0; objIdx < job.numObjs; objIdx++) {
        clusterIdx = membership[objIdx];
        for (columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            distToCenter[clusterIdx][columnIdx] += powf(matrix[objIdx][columnIdx] - scales[clusterIdx][columnIdx], 2);
        }
    }

    matrixFree(scales, numClusters);

    return getScaleSum(distToCenter, clusterCount, clusterWeights, pValue, job);
}


float *subtractLists(const float *list, const float *subtrahend, long numColumns) {
    /* Subtract two lists of same size value by value */

    float *result = floatArrayAlloc(numColumns, 0);

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result[columnIdx] = list[columnIdx] - subtrahend[columnIdx];
    }

    return result;
}


float *divideLists(const float *list, float divisor, long numColumns) {
    /* Divide two lists of same size value by value */

    float *result = floatArrayAlloc(numColumns, 0);

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result[columnIdx] = list[columnIdx] / divisor;
    }

    return result;
}


float listToNumber(float *vector, const long numColumns) {
    /* Reduce list to number: sqrtf(sum(list^2)) */

    float result = 0;

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result += powf(vector[columnIdx], 2);
    }

    return sqrtf(result);
}


float absoluteSum(const float *list, long numColumns) {
    /* Return sum of absolute values from list */

    float result = 0;

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (list[columnIdx] > 0) {
            result += list[columnIdx];
        } else {
            result -= list[columnIdx];
        }
    }
    return result;
}


float maxAbsolute(float *vector, long numColumns) {
    /* Max absolute number in list */

    float ans = FLT_MIN;

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
    float value;

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {

        value = fabsf(vector[columnIdx]) - lam;
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


void updatesWeights(const float **matrix, const mtxDimensions jobInfo, int *membership, float l1Bound,
                    float *clusterWeights, float pValue, float *weights) {
    /* Main routine to update column weights */
    printf("Performing weights update...");

    const long numColumns = jobInfo.numColumns;

    float *wcssPerFeature = getWcss(matrix, jobInfo, membership, clusterWeights, pValue);

    int *arrayOfOnes = getArrayOfOnes(jobInfo.numObjs);
    float *tssPerFeature = getWcss(matrix, jobInfo, arrayOfOnes, NULL, pValue);
    free(arrayOfOnes);

    float *weightsDifference = subtractLists(tssPerFeature, wcssPerFeature, numColumns);

    float lam = calculateLam(weightsDifference, l1Bound, numColumns);

    float *weightsUnscaled = soft(weightsDifference, lam, numColumns);

    float *resultWeights = divideLists(weightsUnscaled, listToNumber(weightsUnscaled, numColumns), numColumns);
    memcpy(weights, resultWeights, sizeof(float) * numColumns);


    free(wcssPerFeature);
    free(tssPerFeature);
    free(weightsDifference);
    free(weightsUnscaled);
    free(resultWeights);

    printf("Done\n");
}
