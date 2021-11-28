#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#include <unistd.h>
#define BLOCK_SZ 256

typedef struct kMeansOutput {
    int flag;
    int *cluster;
    float *centers;
    float *clusterWeights;
    float pValue;
} kMeansOutput;


// Struct with matrices dimensions
typedef struct mtxDimensions {
    const long numObjs;
    const long cpuNumObjs;
    const long d_numObjs;
    const long numColumns;
    const int numClusters;
    const int fit;
} mtxDimensions;


void getJobTimes(struct timespec init, struct timespec reading, struct timespec total) {
    /* Get times spent */

    float readingTime = (reading.tv_sec - init.tv_sec);
    readingTime += (reading.tv_sec - init.tv_sec) / 1000000000.0;
    printf("\nReading Time: %f \n", readingTime);

    float executionTime = (total.tv_sec - init.tv_sec);
    executionTime += (total.tv_nsec - init.tv_nsec) / 1000000000.0;
    printf("Total execution time is: %f\n", executionTime);
}



float *floatArrayAlloc(long size, int populate) {
    float *array;
    if (populate) {
        array = (float *) calloc(1, sizeof(float) * size);
    } else {
        array = (float *) malloc(sizeof(float) * size);
    }
    if (array == NULL) exit(EXIT_FAILURE);

    return array;
}

int *intArrayAlloc(long size, int populate) {
    int *array;
    if (populate) {
        array = (int *) calloc(1, sizeof(int) * size);
    } else {
        array = (int *) malloc(sizeof(int) * size);
    }
    if (array == NULL) exit(EXIT_FAILURE);

    return array;
}

float *cudaFloatAlloc(long arrSize, int populate) {
    float *array;

    cudaMalloc(&array, sizeof(float) * arrSize);

    if (populate) {
        cudaMemset(array, 0, sizeof(float) * arrSize);
    }

    return array;
}

int *cudaIntAlloc(long arrSize, int populate) {
    int *array;

    cudaMalloc(&array, sizeof(int) * arrSize);

    if (populate) {
        cudaMemset(array, 0, sizeof(int) * arrSize);
    }

    return array;
}

void printArray(const float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 30) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%f ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}

void printIntArray(const int arr[], int size) {
    /* Display int array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 15) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%d ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}

void debugFloat(float *target, long size) {
    float *debugArr = floatArrayAlloc(size, 0);
    cudaMemcpy(debugArr, target, sizeof(float) * size, cudaMemcpyDefault);
    printArray(debugArr, size);
    exit(1);
}

void debugInt(int *target, long size) {
    int *debugArr = intArrayAlloc(size, 0);
    cudaMemcpy(debugArr, target, sizeof(int) * size, cudaMemcpyDefault);
    printIntArray(debugArr, size);
    exit(1);
}

FILE *openMatrixFile(char *fileName) {
    /* Reads matrix from mtx file */

    //fileName = (char *) "C:\\Users\\Tango\\Final\\data\\teste.txt";

    FILE *mtxFile = fopen(fileName, "r");
    if (mtxFile == NULL) {
        fprintf(stderr, "Fatal: failed to open arg file: %s!\n", fileName);
        exit(EXIT_FAILURE);
    }
    return mtxFile;
}


void getMatrixDimensions(FILE *mtxFile, long *rows, long *columns) {
    /* Get matrix dimensions from file */

    char *line = NULL;
    char *stringNumber;
    char *ptr;
    size_t len = 0;

    getline(&line, &len, mtxFile);  // Discard first line
    getline(&line, &len, mtxFile);  // Read line with dimensions

    stringNumber = strtok(line, " ");
    *columns = strtol(stringNumber, &ptr, 10);

    stringNumber = strtok(NULL, " ");
    *rows = strtol(stringNumber, &ptr, 10);

    printf("\nNumber of cells: %ld\nNumber of genes: %ld\n", *rows, *columns);
    free(line);
}


void populateMatrix(FILE *argFile, float *matrix, long numColumns, float *sum) {
    /* Assign values to the matrix */

    char *line = NULL;
    char *stringNumber;
    size_t len = 0;

    long row, column;
    float value;
    char *ptr;

    printf("\nPopulating matrix... ");
    while (getline(&line, &len, argFile) != -1) {

        stringNumber = strtok(line, " ");
        column = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        row = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        value = strtof(stringNumber, &ptr);

        matrix[(row * numColumns) + column] = value;
        sum[row] += value;
    }
    free(line);

    printf("Done!\n");
}

void writeOutput(float *centers, int *membership, const mtxDimensions job, const float *columnsWeights, const int nonZeroWeights) {
    /* Write results to output file */

    FILE *centFiles = fopen("center.txt", "w+");
    if (centFiles == NULL) {
        fprintf(stderr, "Fatal: failed to open output file.\n");
        exit(EXIT_FAILURE);
    }

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        long centersCol = 0;

        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            if (columnsWeights[columnIdx]) {
                fprintf(centFiles, "%f ", centers[clusterIdx * nonZeroWeights + centersCol]);
                centersCol++;
            } else {
                fprintf(centFiles, "%f ", 0.0f);
            }
        }
        fprintf(centFiles, "\n");
    }
    fclose(centFiles);

    FILE *outFile = fopen("members.txt", "w+");
    if (outFile == NULL) {
        fprintf(stderr, "Fatal: failed to open output file.\n");
        exit(EXIT_FAILURE);
    }

    for (int arrayIndex = 0; arrayIndex < job.numObjs; arrayIndex++) {
        fprintf(outFile, "%d\n", membership[arrayIndex]);
    }


    fclose(outFile);
}

void checkGPUMem(const long numObjs, const long numColumns, long *d_numObjs, long *d_arrSize, int *fit) {
    const long arrSize = numObjs * numColumns;

    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);

    double free_m, total_m, used_m;
    free_m = free_t / 1048576.0;
    total_m = total_t / 1048576.0;
    used_m = total_m - free_m;

    printf("GPU mem free %.2f MB. Mem total %.2f MB. Mem used %.2f MB\n", free_m, total_m, used_m);

    if ((double) (sizeof(float) * arrSize) > 0.9f * free_t) {
        printf("\nMatrix doesn't fit in GPU memory\n");

        double total_percentage = (double) ((sizeof(float) * arrSize) * 100) / free_t;

        *d_numObjs = (long) (.9 * numObjs);

        *d_arrSize = numColumns * *d_numObjs;
        *fit = 0;


        printf("Will fit %ld from %f \n\n", *d_numObjs, total_percentage);
    } else {
        printf("\nMatrix fit in memory!\n");
        *d_arrSize = arrSize;
        *d_numObjs = numObjs;
        *fit = 1;
    }
}


/* ---------------------------------------  Update Weights  --------------------------------------- */


int *getArrayOfOnes(const long numObjs) {
    /* Initialize an array of ones */

    int *allOneArray = intArrayAlloc(numObjs, 0);
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        allOneArray[objIdx] = 1;
    }

    return allOneArray;
}


void cpuGetScales(const float *matrix, const mtxDimensions job, const int *membership, float *scales) {


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
    // Loop over objects to get sum of points

    int columnIdx = blockIdx.x * blockDim.x + threadIdx.x;


    if (columnIdx < job.numColumns) {
        for (int objIdx = 0; objIdx < job.d_numObjs; objIdx++) {
            int clusterIdx = d_membership[objIdx];
            d_scales[(clusterIdx * job.numColumns) + columnIdx] += d_matrix[(objIdx * job.numColumns) + columnIdx];
        }
    }
}


void resolveScales(const float *matrix, const float *d_matrix, const mtxDimensions job, const int  *membership,
                   const int* d_membership, float *d_scales) {
    const long numClusters = job.numClusters, numColumns = job.numColumns;

    d_getScales<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ>>>(d_matrix, job, d_membership, d_scales);

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
    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        if (d_membership[objIdx] == clusterIdx) {
            d_clusterCount[clusterIdx]++;
        }
    }
}


__global__
void d_getScalesMean(float *d_scales, int const *d_clusterCount, const long numColumns) {
    // Loop over clusters to get mean of points
    int columnIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;

    d_scales[(clusterIdx * numColumns) + columnIdx] =
            d_scales[(clusterIdx * numColumns) + columnIdx] / (float) d_clusterCount[clusterIdx];
}



void getScalesMean (float *scales, const int *clusterCount, const mtxDimensions job) {

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            scales[(clusterIdx * job.numColumns) + columnIdx] /=  (float) clusterCount[clusterIdx];
        }
    }
}


void cpuGetWeightedDistances(const float *matrix, const float *scales, float *distToCenter, const int *membership,
                             const mtxDimensions job) {
    const long numColumns = job.numColumns;

    // Loop over objects to get sum of distances

    for (int objIdx = job.d_numObjs; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            distToCenter[(clusterIdx * numColumns) + columnIdx] += powf(
                    matrix[(objIdx * numColumns) + columnIdx]
                    - scales[(clusterIdx * numColumns) + columnIdx], 2);
        }

    }
}


__global__
void d_getWeightedDistances(const float *d_matrix, const float *d_scales, float *d_distToCenter, const int *d_membership,
                            const mtxDimensions job) {
    // Loop over clusters to get mean of points
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
    const long numColumns = job.numColumns, numClusters = job.numClusters;

    d_getWeightedDistances <<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> > (d_matrix, d_scales, d_distToCenter, d_membership, job);

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

float * solveWcss (const float *distToCenter, const int *clusterCount, const float *clusterWeights,
                   const float pValue, const mtxDimensions job) {

    const long numColumns = job.numColumns;
    float *wcss = floatArrayAlloc(numColumns, 1);

#pragma omp parallel for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
            if (clusterCount[clusterIdx] > 1) {
                if (clusterWeights) {
                    wcss[columnIdx] +=
                            distToCenter[(clusterIdx * numColumns) + columnIdx] *
                            powf(clusterWeights[clusterIdx], pValue);
                } else {
                    wcss[columnIdx] += distToCenter[(clusterIdx * numColumns) + columnIdx];
                }
            }
        }
    }

    return wcss;
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
    cudaMemcpy(d_membership, membership, sizeof(float) *  job.numObjs, cudaMemcpyDefault);

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
    d_solveWcss<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> >(d_distToCenter, d_clusterCount, d_clusterWeights, d_wcss, pValue, job);


    cudaFree(d_scales);
    cudaFree(d_distToCenter);
    cudaFree(d_clusterCount);
    return d_wcss;
}


float *subtractLists(const float *list, const float *subtrahend, long numColumns) {
    /* Subtract two lists of same size value by value */

    float *result = floatArrayAlloc(numColumns, 0);

#pragma omp parallel for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        result[columnIdx] = list[columnIdx] - subtrahend[columnIdx];
    }

    return result;
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


void updatesWeights(const float *matrix, const float *d_matrix, const mtxDimensions jobInfo, int *membership, float l1Bound,
                    float *clusterWeights, float pValue, float *weights) {
    /* Main routine to update column weights */
    printf("Performing weights update...");

    float *d_clusterWeights = cudaFloatAlloc(jobInfo.numClusters, 0);
    cudaMemcpy(d_clusterWeights, clusterWeights, sizeof(float) *  jobInfo.numClusters, cudaMemcpyDefault);

    const long numColumns = jobInfo.numColumns;

    float *d_wcssPerFeature = getWcss(matrix, d_matrix, jobInfo, membership, d_clusterWeights, pValue);

    int *arrayOfOnes = getArrayOfOnes(jobInfo.numObjs);
    float *d_tssPerFeature = getWcss(matrix, d_matrix, jobInfo, arrayOfOnes, NULL, pValue);
    free(arrayOfOnes);


    float *d_weightsDifference = cudaFloatAlloc(numColumns, 0);
    d_subtractLists<<<(numColumns + 255) / BLOCK_SZ, BLOCK_SZ >> > (d_weightsDifference, d_tssPerFeature, d_wcssPerFeature, numColumns);

    float *weightsDifference = floatArrayAlloc(numColumns, 0);
    cudaMemcpy(weightsDifference, d_weightsDifference, sizeof(float) * numColumns , cudaMemcpyDefault);

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


/* ---------------------------------------  MinMax KMeans  --------------------------------------- */

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


void initializeClusterWeights(float *clusterWeights, float *oldClusterWeights, const long numClusters) {
    /* Initialize cluster weights */

    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = 1 / (float) numClusters;
        oldClusterWeights[clusterIdx] = 1 / (float) numClusters;
    }
}

__global__
void d_initializeClusterWeights(float *d_clusterWeights, float *d_oldClusterWeights, const long numClusters) {
    int clusterIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (clusterIdx < numClusters) {
        d_clusterWeights[clusterIdx] = 1 / (float) numClusters;
        d_oldClusterWeights[clusterIdx] = 1 / (float) numClusters;
    }
}

__global__
void d_getEuclideanDist(const float *d_matrix, float *d_centroids, float *d_euclideanDistances,
                        const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters */

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
    /* Find euclidean distances for every attribute in every object to each clusters */

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

    d_getEuclideanDist<<<job.d_numObjs, job.numClusters >>> (d_matrix, d_centroids, d_euclideanDistances, job);

    if(!job.fit) {
        cudaMemcpy(centroids, d_centroids, sizeof(float) * job.numClusters * job.numColumns, cudaMemcpyDefault);
        float *tempEuclideanDistances = floatArrayAlloc(job.cpuNumObjs * job.numClusters, 1);
        cpuGetEuclideanDist(matrix, centroids, tempEuclideanDistances, job); // pode rodar junto com o cuda
        cudaMemcpy(d_euclideanDistances + job.d_numObjs * job.numClusters, tempEuclideanDistances,
                   sizeof(float) * job.cpuNumObjs * job.numClusters, cudaMemcpyDefault);
        free(tempEuclideanDistances);

    }

}

void initializeDistanceWeights (float *weightedDistances, const float *clusterWeights, const  float *euclideanDistances,
                                const mtxDimensions job, const float pValue) {
    // Initialize matrix weights
    const long numObjs = job.numObjs, numClusters = job.numClusters;

#pragma omp parallel for collapse(2)
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx + clusterIdx * numObjs] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx + clusterIdx * numObjs];
        }
    }
}

__global__
void d_initializeDistanceWeights(float *d_weightedDistances, const float *d_clusterWeights,
                                 const float *d_euclideanDistances, const mtxDimensions job, const float pValue) {

    int objIdx = blockIdx.x;
    int clusterIdx = threadIdx.x;


    d_weightedDistances[objIdx + clusterIdx * job.numObjs] =
            powf(d_clusterWeights[clusterIdx], pValue) * d_euclideanDistances[objIdx + clusterIdx * job.numObjs];

}

__global__
void d_findNearestCluster(const float *d_distances, float *d_objsMtxMembership, int *d_objsMembership, const mtxDimensions job) {
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
    for (int cluster = 0; cluster < job.numClusters; cluster++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            if (d_objsMembership[objIdx] == cluster)
                d_clusterSize[cluster]++;
        }
    }
}


int *getObjectsDistribution(int *d_clusterSize, const float *d_weightedDistances, float *d_objsMtxMembership,
                            int *d_objsMembership, const mtxDimensions job) {

    const long numObjs = job.numObjs, numClusters = job.numClusters;
    int *clusterSize = intArrayAlloc(numClusters, 1);


    d_findNearestCluster<<<(numObjs + 255) / BLOCK_SZ, BLOCK_SZ>>> (d_weightedDistances, d_objsMtxMembership, d_objsMembership, job);


    d_getClusterSize<<<1,1>>>(d_objsMembership, d_clusterSize, job);
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
void d_getClusterMeans(const float *d_matrix, float *d_centroids, const int *d_objsMembership, const mtxDimensions job) {
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
                         const int *d_clusterSize, int *objsMembership, int *d_objsMembership, const mtxDimensions job) {

    const long numClusters = job.numClusters, numColumns = job.numColumns;


    d_getClusterMeans<<< job.numColumns, job.numClusters >>>(d_matrix, d_centroids, d_objsMembership, job);

    if (!job.fit) {
        cudaMemcpy(objsMembership, d_objsMembership, sizeof(int) * job.numObjs, cudaMemcpyDefault);
        cudaMemcpy(centroids, d_centroids, sizeof(float) * numClusters * numColumns, cudaMemcpyDefault);
        cpuClusterMeans(matrix, centroids, objsMembership, job); // rodar após
        cudaMemcpy(d_centroids, centroids, sizeof(float) * numClusters * numColumns, cudaMemcpyHostToDevice);
    }



    d_setMeans<<< job.numColumns, job.numClusters >>> (d_centroids, d_clusterSize, job);
}


void updateClusterWeights(float *clusterWeights, const float *objsMtxMembership, const float *euclideanDistances,
                          const mtxDimensions job, const float pValue) {
    /* Updates cluster weights according to distances and size */

    const long numClusters = job.numClusters;
    float sumWeightedDistances = 0;
    float *clustersDistance = floatArrayAlloc(numClusters, 1);

    // Loop over clusters to get sum of distances for each obj
#pragma omp parallel for reduction(+:sumWeightedDistances)
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            clustersDistance[clusterIdx] +=
                    objsMtxMembership[(objIdx * numClusters) + clusterIdx] * euclideanDistances[(objIdx * numClusters) + clusterIdx];
        }
        sumWeightedDistances += powf(clustersDistance[clusterIdx], (1 / (1 - pValue)));
    }

    // Loop over sum of clusters to update weights
#pragma omp parallel for
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = powf(clustersDistance[clusterIdx], (1 / (1 - pValue))) / sumWeightedDistances;
    }


    free(clustersDistance);
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
            sumWeights+= weightedDistances[objIdx * numClusters + clusterIdx];
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
    if(jobInfo.numClusters == 2) {
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
    d_initializeClusterWeights<<<1, numClusters>>>(d_clusterWeights, d_oldClusterWeights, numClusters);



    // Initialize euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 1);
    float *d_euclideanDistances = cudaFloatAlloc(numObjs * numClusters, 0);
    resolveEuclideanDistances(matrix, d_matrix, centroids, d_centroids, d_euclideanDistances, jobInfo);


    // Initialize matrix weights
    float *weightedDistances = floatArrayAlloc(numObjs * numClusters, 0);
    float *d_weightedDistances = cudaFloatAlloc(numObjs * numClusters, 0);
    d_initializeDistanceWeights <<< numObjs, numClusters >> > (d_weightedDistances, d_clusterWeights, d_euclideanDistances, jobInfo, pValue);


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
        int *clusterSize = getObjectsDistribution(d_clusterSize, d_weightedDistances, d_objsMtxMembership, d_objsMembership, jobInfo);

        // Lower pValue if some cluster is empty
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                cudaMemcpy(d_objsMtxMembership, d_oldObjsMembership, sizeof(float) * numObjs * numClusters, cudaMemcpyDefault);
                cudaMemcpy(d_clusterWeights, d_oldClusterWeights, sizeof(float) *  numClusters, cudaMemcpyDefault);
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
        resolveClusterMeans(matrix, d_matrix, centroids, d_centroids, d_clusterSize, objsMembership, d_objsMembership, jobInfo);


        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            cudaMemcpy(d_oldObjsMembership, d_objsMtxMembership, sizeof(float) * numObjs * numClusters, cudaMemcpyDefault);
            cudaMemcpy(d_oldClusterWeights, d_clusterWeights, sizeof(float) *  numClusters, cudaMemcpyDefault);

            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        resolveEuclideanDistances(matrix, d_matrix, centroids, d_centroids, d_euclideanDistances, jobInfo);

        // Update clusters weights according to distances
        d_updateClusterWeights<<<1, numClusters >> >(d_clusterWeights, d_objsMtxMembership, d_euclideanDistances, jobInfo, pValue); //ok

        // Update sum of weights to verify condition
        oldSumWeights = sumWeights;

        cudaMemcpy(euclideanDistances, d_euclideanDistances, sizeof(float) * (numObjs * numClusters), cudaMemcpyDefault);
        cudaMemcpy(clusterWeights, d_clusterWeights, sizeof(float) *  numClusters, cudaMemcpyDefault);
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


/* ---------------------------------------  Update Centers  --------------------------------------- */


void getWeightedObjects(const float *matrix, float *weightedObj, long numObjs, long numColumns, const float *weights,
                        const long nonZeroWeights) {
    /* Calculates weight for every column in every object */

    long numRows = 0;

#pragma omp for
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (weights[columnIdx] != 0) {
            for (int objIdx = 0; objIdx < numObjs; objIdx++) {
                weightedObj[(objIdx * nonZeroWeights) + numRows] =
                        matrix[(objIdx * numColumns) + columnIdx] * sqrtf(weights[columnIdx]);
            }
            numRows++;
        }
    }
}


void moveCenterMeanBalanced(const float *weightedObjs, float *balancedCenters, const mtxDimensions job,
                            long nonZeroWeights, const int *centroids) {
    /* Moves centers according to calculated weights */

    int cluster, columnIdx;

    int *sumClusterPoints = intArrayAlloc(job.numClusters, 1);



    // Get sum of points in cluster considering weights
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = centroids[objIdx];
        sumClusterPoints[cluster]++;

#pragma omp parallel for firstprivate(objIdx, cluster)
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[(cluster * nonZeroWeights) + columnIdx] += weightedObjs[(objIdx * nonZeroWeights) +
                                                                                    columnIdx];
        }
    }

    // Calculate mean for each cluster and move center to it
#pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[(clusterIdx * nonZeroWeights) + columnIdx] =
                    balancedCenters[(clusterIdx * nonZeroWeights) + columnIdx] / (float) sumClusterPoints[clusterIdx];
        }
    }

    free(sumClusterPoints);
}


int countUniques(const int *uniqueNearest, int numClusters) {
    // Return count of unique values in array

    int ans = 0;
#pragma omp parallel for reduction(+:ans)
    for (int i = 0; i < numClusters; i++) {
        if (uniqueNearest[i] > 0) {
            ans++;
        }
    }

    return ans;
}


void cpuFindNearestCluster(int *clustersMembership, const float *euclideanDistances, const mtxDimensions job) {
    /* Returns the  nearest cluster from the point */
    const long numClusters = job.numClusters;


    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {

        int nearestCluster = 0;
        float min_dist = FLT_MAX;

        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (min_dist > euclideanDistances[(objIdx * numClusters) + clusterIdx]) {
                min_dist = euclideanDistances[(objIdx * numClusters) + clusterIdx];
                nearestCluster = clusterIdx;
            }
        }
        clustersMembership[nearestCluster]++;
    }
}


kMeansOutput updateCenters(const float *matrix, mtxDimensions jobInfo, float *columnsWeights,
                           int *membership, int nonZeroWeights) {
    /* Main routine for updating centers according to objects weights */

    printf("Performing centers update...");
    const long numObjs = jobInfo.numObjs, numColumns = jobInfo.numColumns, numClusters = jobInfo.numClusters;

    float *weightedObjs = floatArrayAlloc(numObjs * nonZeroWeights, 0);
    float *balancedCenters = floatArrayAlloc(numClusters * nonZeroWeights, 1);

    // Weights every object
    getWeightedObjects(matrix, weightedObjs, numObjs, numColumns, columnsWeights, nonZeroWeights);

    // Move centers according to weights
    moveCenterMeanBalanced(weightedObjs, balancedCenters, jobInfo, nonZeroWeights, membership);


    mtxDimensions updateJob = {numObjs, numObjs, 0, nonZeroWeights, numClusters, 0};

    // Get euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 0);
    cpuGetEuclideanDist((const float *) weightedObjs, balancedCenters, euclideanDistances, updateJob);

    // Get nearest cluster for each point
    int *clustersMembership = intArrayAlloc(numClusters, 1);
    cpuFindNearestCluster(clustersMembership, euclideanDistances, updateJob);

    free(euclideanDistances);

    // Perform MinMaxKMeans with defined centers if every cluster is populated else random
    kMeansOutput kMeansResult;
    if (countUniques(clustersMembership, numClusters) == numClusters) {
        kMeansResult = getKMeansResult(weightedObjs, NULL, balancedCenters, updateJob, 0);
    } else {
        kMeansResult = getKMeansResult(weightedObjs, NULL, balancedCenters, updateJob, 1);
    }

    free(weightedObjs);
    free(clustersMembership);

    printf("Done\n");
    return kMeansResult;

}


/* ---------------------------------------  Sparse KMeans  --------------------------------------- */


void initializeWeights(float *columnsWeights, const long numColumns) {
    /* initialize weights array to 1/sqrtf(columns) */

    float initWeight = 1.f / sqrtf((float) numColumns);
#pragma omp parallel for
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        columnsWeights[columnIndex] = initWeight;
    }
}

float absoluteSubtract(const float *weights, const float *weights_old, const long numColumns) {
    /* Gets the sum of the difference between two arrays */

    float result = 0;
#pragma omp parallel for reduction(+:result)
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        result += fabsf(weights[columnIndex] - weights_old[columnIndex]);
    }

    return result;
}


int getNonZeroWeights(const float *columnWeights, const long numColumns) {
    /* Get number of non zero weight values */

    int nonZero = 0;
#pragma omp parallel for reduction(+:nonZero)
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (columnWeights[columnIdx] != 0) nonZero++;
    }

    return nonZero;
}


kMeansOutput sparseKMeans(const float *matrix, const mtxDimensions jobInfo, float *columnsWeights, const int maxLoops) {
    /* Main routine for sparse KMeans */

    // Initialize values
    const long numColumns = jobInfo.numColumns;
    float boundWeight = 2.f;

    float *d_matrix = cudaFloatAlloc(jobInfo.d_numObjs * numColumns, 0);
    cudaMemcpy(d_matrix, matrix, sizeof(float) * jobInfo.d_numObjs * numColumns, cudaMemcpyDefault);


    // Initialize columns weights
    float *oldColumnsWeights = floatArrayAlloc(numColumns, 0);
    initializeWeights(columnsWeights, numColumns);


    // Perform MinMaxKMeans once
    float *centroids = floatArrayAlloc(jobInfo.numClusters * numColumns, 1);

    double kMeansStart = omp_get_wtime();
    kMeansOutput kMeansResult = getKMeansResult(matrix, d_matrix, centroids, jobInfo, 1);
    double kMeansEnd = omp_get_wtime();

    // Variables for loop
    int nonZeroWeights = numColumns, nIter = 1;
    float condition = 1;
    const float threshold = 0.0001f;

    double totalCenters = 0, totalWeigths = 0, elapsedStart;

    // Loop while condition not reached
    while (condition > threshold && nIter < maxLoops) {
        printf("\nRunning loop %d\n", nIter);
        printf("Weights convergence: %f\n ", condition);
        printf("\n");


        memcpy(oldColumnsWeights, columnsWeights, sizeof(float) * numColumns);

        // Updates centers according to cluster weights
        if (nIter > 1) {

            free(kMeansResult.clusterWeights);
            free(kMeansResult.centers);
            elapsedStart = omp_get_wtime();
            kMeansResult = updateCenters(matrix, jobInfo, columnsWeights, kMeansResult.cluster, nonZeroWeights);
            totalCenters += omp_get_wtime() - elapsedStart;

        }

        // Updates columns weights
        elapsedStart = omp_get_wtime();
        updatesWeights(matrix, d_matrix, jobInfo, kMeansResult.cluster, boundWeight, kMeansResult.clusterWeights,
                       kMeansResult.pValue, columnsWeights);
        totalWeigths += omp_get_wtime() - elapsedStart;
        nonZeroWeights = getNonZeroWeights(columnsWeights, numColumns);


        // Get condition for next loop
        condition = absoluteSubtract(columnsWeights, oldColumnsWeights, numColumns) /
                    absoluteSum(oldColumnsWeights, numColumns);
        nIter++;
    }

    free(oldColumnsWeights);

    printf("Kmeans Time = %f\n", kMeansEnd-kMeansStart);
    printf("Centers Time = %f\n", totalCenters);
    printf("Weights Time = %f\n", totalWeigths);

    return kMeansResult;
}


void normalizeCPM(float *matrix, const float *sum, long numRows, long numColumns) {
    /* Perform CPM normalization */

    printf("\nNormalizing...\n");

#pragma omp parallel for collapse(2)
    for (int objIdx = 0; objIdx < numRows; objIdx++) {
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            matrix[(objIdx * numColumns) + columnIdx] = (matrix[(objIdx * numColumns) + columnIdx] / sum[objIdx]) * 1e6f;
        }
    }
}


int main(int argc, char *argv[]) {

    // Initialize times
    struct timespec initTime, readingTime, totalTime;
    clock_gettime(CLOCK_MONOTONIC, &initTime);

    // Open matrix file
    FILE *mtxFile = openMatrixFile(argv[1]);

    // Reads number of rows and columns from file
    long numObjs, numColumns;
    getMatrixDimensions(mtxFile, &numObjs, &numColumns);
    long arrSize = numColumns * numObjs;


    // Populate matrix and expression sum for normalization
    float *rowSum = floatArrayAlloc(numObjs, 1);
    float *matrix = floatArrayAlloc(arrSize, 1);
    populateMatrix(mtxFile, matrix, numColumns, rowSum);
    clock_gettime(CLOCK_MONOTONIC, &readingTime);


    // Normalize matrix using CPM
    if (!strcmp(argv[3], "yes")) {
        double kMeansStart = omp_get_wtime();
        normalizeCPM(matrix, rowSum, numObjs, numColumns);
        printf("Normalization Time = %f\n", omp_get_wtime() - kMeansStart);
    }
    free(rowSum);


    // Check GPU memory
    long d_numObjs, d_arrSize;
    int fit;
    checkGPUMem(numObjs, numColumns, &d_numObjs, &d_arrSize, &fit);


    // Initalize values for run
    long numClusters = 7;
    if (numObjs < 10) numClusters = 2;
    const int maxLoops = 10;
    const int numThreads = atoi(argv[2]);
    printf("Num Threads = %d\n", numThreads);
    omp_set_num_threads(numThreads);

    const mtxDimensions jobInfo = {numObjs, numObjs - d_numObjs, d_numObjs, numColumns, numClusters, fit};
    float *columnsWeights = floatArrayAlloc(numColumns, 1);


    // Perform Sparse Min Max KMeans
    puts("\nInitializing Sparse KMeans");
    kMeansOutput kMeansResult = sparseKMeans((const float *) matrix, jobInfo, columnsWeights, maxLoops);
    clock_gettime(CLOCK_MONOTONIC, &totalTime);
    puts("\nDone!");


    // Writes output
    writeOutput(kMeansResult.centers, kMeansResult.cluster, jobInfo, columnsWeights, getNonZeroWeights(columnsWeights, numColumns));


    // Get job time
    getJobTimes(initTime, readingTime, totalTime);


    free(matrix);
    free(kMeansResult.centers);
    free(kMeansResult.cluster);
    free(kMeansResult.clusterWeights);
    free(columnsWeights);

    return 0;
}
