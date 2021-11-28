#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

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
    long numColumns;
    const int numClusters;
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


void printArray(const float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
//        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 30) {
//            printf(". . . . ");
//            arrayIndex = size - 5;
//        }

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


FILE *openMatrixFile(char *fileName) {
    /* Reads matrix from mtx file */

//    fileName = (char *) "C:\\Users\\Tango\\Final\\data\\teste.txt";

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


void populateMatrix(FILE *argFile, float *matrix, long numColumns, const int numObjs, float *sum) {
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

void checkGPUMem(const long numObjs, const long numColumns, long *d_numObjs, long *d_arrSize, int *fit, int teste) {
    const long arrSize = numObjs * numColumns;

//    size_t free_t, total_t;
//    cudaMemGetInfo(&free_t, &total_t);

    size_t free_t =1, total_t=1;
    double free_m, total_m, used_m;
    free_m = free_t / 1048576.0;
    total_m = total_t / 1048576.0;
    used_m = total_m - free_m;

    printf("mem free %.2f MB mem total %.2f MB mem used %.2f MB\n", free_m, total_m, used_m);

    if ((double) (sizeof(float) * arrSize) > 0.9f * free_t || teste) {
        printf("\nDoesn't fit here\n");

        double total_percentage = (double) ((sizeof(float) * arrSize) * 100) / free_t;

        if (!teste) *d_numObjs = (long) ((90 * numObjs) / total_percentage);
        else *d_numObjs = 1;

        *d_arrSize = numColumns * *d_numObjs;
        *fit = 0;


        printf("Will fit %ld from %f \n\n", *d_numObjs, total_percentage);
    } else {
        printf("\nFit gostoso demais\n");
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

    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            scales[(clusterIdx * job.numColumns) + columnIdx] += matrix[(objIdx * job.numColumns) + columnIdx];
        }
    }
}


void resolveScales(const float *matrix, const mtxDimensions job, const int  *membership, float *scales, int *clusterCount) {

    cpuGetScales(matrix, job, membership, scales); // rodar após



    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
        clusterCount[clusterIdx]++;
    }

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

    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = membership[objIdx];
#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            distToCenter[(clusterIdx * numColumns) + columnIdx] += powf(
                    matrix[(objIdx * numColumns) + columnIdx]
                    - scales[(clusterIdx * numColumns) + columnIdx], 2);
        }
    }
}

void resolveWeightedDistances(const float *matrix, const float *scales, float *distToCenter,
                              const int *membership, const mtxDimensions job) {

    cpuGetWeightedDistances(matrix, scales, distToCenter, membership, job); // rodar após

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


float *getWcss(const float *matrix, const mtxDimensions job, const int *membership,
               float *clusterWeights, float pValue) {
    /* Get sum of distances of every column to calculate weights*/

    const long numColumns = job.numColumns, numClusters = job.numClusters;

    float *scales = floatArrayAlloc(numClusters * numColumns, 1);
    float *distToCenter = floatArrayAlloc(numClusters * numColumns, 1);
    int *clusterCount = intArrayAlloc(job.numClusters, 1);

    resolveScales(matrix, job, membership, scales, clusterCount);

    getScalesMean(scales, clusterCount, job);


    resolveWeightedDistances(matrix, scales, distToCenter, membership, job);

    // Loop over objects to get sum of distances
    float *wcss = solveWcss(distToCenter, clusterCount, clusterWeights, pValue, job);


    free(scales);
    free(distToCenter);
    free(clusterCount);
    return wcss;
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


void updatesWeights(const float *matrix, const mtxDimensions jobInfo, int *membership, float l1Bound,
                    float *d_clusterWeights, float pValue, float *weights) {
    /* Main routine to update column weights */
    printf("Performing weights update...");

    const long numColumns = jobInfo.numColumns;

    float *wcssPerFeature = getWcss(matrix, jobInfo, membership, d_clusterWeights, pValue);

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



void cpuGetEuclideanDist(const float *matrix, float *centroids, float *euclideanDistances, const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters */

    const long numColumns = job.numColumns;


    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {

#pragma omp parallel for
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            float distance = 0;
            for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
                distance += powf((matrix[(objIdx * numColumns) + columnIdx] -
                                  centroids[(clusterIdx * numColumns) + columnIdx]),
                                 2);
            }

            euclideanDistances[(objIdx  * job.numClusters) + clusterIdx] = sqrtf(distance);
        }
    }

}

void resolveEuclideanDistances(const float *matrix,float *centroids, float *euclideanDistances, const mtxDimensions job) {

    cpuGetEuclideanDist(matrix, centroids, euclideanDistances, job);

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

int *getObjectsDistribution(const float *weightedDistances, float *objsMtxMembership, int *objsMembership, const mtxDimensions job) {
    const long numObjs = job.numObjs, numClusters = job.numClusters;
    int *clusterSize = intArrayAlloc(numClusters, 1);

#pragma omp parallel for
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        int nearestCluster;
        float min_dist = FLT_MAX;

        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (min_dist > weightedDistances[objIdx * numClusters + clusterIdx]) {
                min_dist = weightedDistances[objIdx * numClusters + clusterIdx];
                nearestCluster = clusterIdx;
            }
        }

        objsMtxMembership[(objIdx * numClusters) + nearestCluster] = 1;
        objsMembership[objIdx] = nearestCluster;

    }


    //#pragma omp for
    for (int cluster = 0; cluster < numClusters; cluster++) {
        for (int objIdx = 0; objIdx < numObjs; objIdx++) {
            if (objsMembership[objIdx] == cluster)
                clusterSize[cluster]++;
        }
    }

    return clusterSize;
}



void cpuClusterMeans(const float *matrix, float *centroids,const int *objsMembership, const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    const long numColumns = job.numColumns;



    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        int clusterIdx = objsMembership[objIdx];

#pragma omp parallel for firstprivate(objIdx, clusterIdx)
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            centroids[(clusterIdx * numColumns) + columnIdx] += matrix[(objIdx * numColumns) + columnIdx];
        }
    }
}



void resolveClusterMeans(const float *matrix, float *centroids, const int *clusterSize,
                         int *objsMembership, const mtxDimensions job) {

#pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[(clusterIdx * job.numColumns) + columnIdx] = 0;
        }
    }

    cpuClusterMeans(matrix, centroids, objsMembership, job);


#pragma omp parallel for collapse(2)
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[(clusterIdx * job.numColumns) + columnIdx] /= (float) clusterSize[clusterIdx];
        }
    }

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


kMeansOutput minMaxKMeans(const float *matrix, float *centroids, const mtxDimensions jobInfo,
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

    printf("Performing MinMaxKMeans...");

    // Initialize matrices and arrays to be used
    float *oldObjsMembership = floatArrayAlloc(numObjs * numClusters, 1);
    float *weightedDistances = floatArrayAlloc(numObjs * numClusters, 1);
    float *objsMtxMembership = floatArrayAlloc(numObjs * numClusters, 1);

    float *clusterWeights = floatArrayAlloc(numClusters, 0);
    float *oldClusterWeights = floatArrayAlloc(numClusters, 0);
    int *objsMembership = intArrayAlloc(numObjs, 0);


    // Initialize cluster weights
    initializeClusterWeights(clusterWeights, oldClusterWeights, numClusters);

    // Initialize euclidean distances
    float *euclideanDistances = floatArrayAlloc(numObjs * numClusters, 1);
    resolveEuclideanDistances(matrix, centroids, euclideanDistances, jobInfo);


    // Initialize matrix weights
    initializeDistanceWeights(weightedDistances, clusterWeights, euclideanDistances, jobInfo, pValue); // ok


    // Variables for loop
    int empty = 0, nIter = 0;
    float oldSumWeights, sumWeights = 0;


    // Loop until weights converge to 0 or hit max loops
    while (1) {
        nIter++;

        // Classify each object and get cluster sizes
        int *clusterSize = getObjectsDistribution(weightedDistances, objsMtxMembership, objsMembership, jobInfo);

        // Lower pValue if some cluster is empty
        for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                memcpy(objsMtxMembership, oldObjsMembership, sizeof(float) * numObjs * numClusters);
                memcpy(clusterWeights, oldClusterWeights, sizeof(float) * numClusters);
                break;
            }
        }

        // Break loop if pValue lesser than 0
        if (pValue < 0) {
            pValue = 0;
            flag = 0;
            free(clusterSize);
            break;
        }

        // Update centroids with distances means
        resolveClusterMeans(matrix, centroids, clusterSize, objsMembership, jobInfo);


        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            memcpy(oldObjsMembership, objsMtxMembership, sizeof(float) * numObjs * numClusters);
            memcpy(oldClusterWeights, clusterWeights, sizeof(float) * numClusters);
            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        resolveEuclideanDistances(matrix, centroids, euclideanDistances, jobInfo);

        // Update clusters weights according to distances
        updateClusterWeights(clusterWeights, objsMtxMembership, euclideanDistances, jobInfo, pValue);

        // Update sum of weights to verify condition
        oldSumWeights = sumWeights;
        sumWeights = updateSumWeights(weightedDistances, euclideanDistances, clusterWeights, pValue, jobInfo);

        free(clusterSize);

        // Break loop when weights convert to 0
        if (nIter > 1) {
            if (fabsf(sumWeights - oldSumWeights) <= 1e-04 || nIter >= 10) {
                break;
            }
        }
    }

    printf("Done\n");

    // Set return results
    kMeansResult.centers = centroids;  // here
    kMeansResult.flag = flag;
    kMeansResult.cluster = objsMembership;  // here
    kMeansResult.clusterWeights = clusterWeights; // here
    kMeansResult.pValue = pValue;

    free(weightedDistances);
    free(objsMtxMembership);
    free(oldObjsMembership);
    free(euclideanDistances);
    free(oldClusterWeights);

    return kMeansResult;
}


void cleanOldKMeans(kMeansOutput oldKMeans) {
    /* Free unused kMeans results */

    free(oldKMeans.cluster);
    free(oldKMeans.clusterWeights);
}


kMeansOutput getKMeansResult(const float *matrix, float *centroids, const mtxDimensions jobInfo,
                             int getRandom) {
    /* Perform MinMax KMeans until all clusters are populated */
    int nIter = 0;

    kMeansOutput kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);

    while (kMeansResult.flag == 0 && nIter < 6) {
        getRandom = 1;
        nIter++;
        cleanOldKMeans(kMeansResult);
        kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);
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


    mtxDimensions updateJob = {numObjs, nonZeroWeights, numClusters};

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
        kMeansResult = getKMeansResult(weightedObjs, balancedCenters, updateJob, 0);
    } else {
        kMeansResult = getKMeansResult(weightedObjs, balancedCenters, updateJob, 1);
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
    const long numColumns = jobInfo.numColumns, numObjs = jobInfo.numObjs;
    float boundWeight = 2.f;


    // Initialize columns weights
    float *oldColumnsWeights = floatArrayAlloc(numColumns, 0);
    initializeWeights(columnsWeights, numColumns);


    // Perform MinMaxKMeans once
    int *membershipArray = intArrayAlloc(numObjs, 0);
    float *centroids = floatArrayAlloc(jobInfo.numClusters * numColumns, 1);

    double kMeansStart = omp_get_wtime();
    kMeansOutput kMeansResult = getKMeansResult(matrix, centroids, jobInfo, 1);
    double kMeansEnd = omp_get_wtime();
    memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);

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
            cleanOldKMeans(kMeansResult);
            free(kMeansResult.centers);
            elapsedStart = omp_get_wtime();
            kMeansResult = updateCenters(matrix, jobInfo, columnsWeights, membershipArray, nonZeroWeights);
            totalCenters += omp_get_wtime() - elapsedStart;
            memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);
        }

        // Updates columns weights
        elapsedStart = omp_get_wtime();
        updatesWeights(matrix, jobInfo, membershipArray, boundWeight, kMeansResult.clusterWeights,
                       kMeansResult.pValue, columnsWeights);
        totalWeigths += omp_get_wtime() - elapsedStart;
        nonZeroWeights = getNonZeroWeights(columnsWeights, numColumns);


        // Get condition for next loop
        condition = absoluteSubtract(columnsWeights, oldColumnsWeights, numColumns) /
                    absoluteSum(oldColumnsWeights, numColumns);
        nIter++;
    }

    free(oldColumnsWeights);
    free(membershipArray);


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
    struct timespec initTime, readingTime, totalTime;
    clock_gettime(CLOCK_MONOTONIC, &initTime);

    const int numThreads = atoi(argv[2]);
    printf("Num threads = %d\n", numThreads);
    omp_set_num_threads(numThreads);

    // Open matrix file
    FILE *mtxFile = openMatrixFile(argv[1]);

    // Reads number of rows and columns from file
    long numObjs, numColumns;
    getMatrixDimensions(mtxFile, &numObjs, &numColumns);
    long arrSize = numColumns * numObjs;


    // Populate matrix and expression sum for normalization
    float *rowSum = floatArrayAlloc(numObjs, 1);
    float *matrix = floatArrayAlloc(arrSize, 1);
    populateMatrix(mtxFile, matrix, numColumns, numObjs, rowSum);
    clock_gettime(CLOCK_MONOTONIC, &readingTime);

    // Normalize matrix using CPM
    if (!strcmp(argv[3], "yes")) {
        double kMeansStart = omp_get_wtime();
        normalizeCPM(matrix, rowSum, numObjs, numColumns);
        printf("Normalization Time = %f\n", omp_get_wtime() - kMeansStart);
    }
    free(rowSum);

    long numClusters = 7;
    if (numObjs < 10) numClusters = 2;
    const int maxLoops = 10;


    const mtxDimensions jobInfo = {numObjs, numColumns, numClusters};
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

//gcc -std=c99 -fopenmp -o cpu_openmp cpu_openmp.c -lm