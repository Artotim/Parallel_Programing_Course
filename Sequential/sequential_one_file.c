#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>


// Struct to hold kMeans results
typedef struct kMeansOutput {
    int flag;
    int *cluster;
    float **centers;
    float *clusterWeights;
    float pValue;
} kMeansOutput;


// Struct with matrices dimensions
typedef struct mtxDimensions {
    const long numObjs;
    const long numColumns;
    const int numClusters;

} mtxDimensions;


/* ______________________________________________ Matrix Array routines _______________________________________________*/

void printArray(float arr[], int size) {
    /* Display float array */

    int arrayIndex = 0;
    while (arrayIndex <= size - 1) {
        if (arrayIndex > 4 && arrayIndex < size - 5 && size > 15) {
            printf(". . . . ");
            arrayIndex = size - 5;
        }

        printf("%f ", arr[arrayIndex]);
        arrayIndex++;
    }
    printf("\n");
}


void printIntArray(int arr[], int size) {
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


void printMatrix(float** matrix, int rows, int columns) {
    /* Prints matrix */

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            printf("%f ",matrix[i][j]);
        }
        putchar('\n');
    }
}


float *floatArrayAlloc(long size, int populate) {
    float *array;
    if (populate) {
        array = calloc(1, sizeof(float) * size);
    } else {
        array = malloc(sizeof(float) * size);
    }
    if(array == NULL) exit(EXIT_FAILURE);

    return array;
}


int *intArrayAlloc(long size, int populate) {
    int *array;
    if (populate) {
        array = calloc(1, sizeof(int) * size);
    } else {
        array = malloc(sizeof(int) * size);
    }
    if(array == NULL) exit(EXIT_FAILURE);

    return array;
}


float **matrixAlloc(int rows, int columns) {
    /* Allocate memory for matrix */

    float **matrix = malloc(sizeof(float) * rows * columns);

    if(matrix == NULL) exit(EXIT_FAILURE);

    for(int i = 0; i < rows; i++) {
        matrix[i] = calloc(1, sizeof(float) * columns);
        if(matrix[i] == NULL) exit(EXIT_FAILURE);
    }

    return matrix;
}


void matrixFree(float **matrix, int rows) {
    /* Free matrix memory */

    for(int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


/* ____________________________________________________ Read Write Routines ___________________________________________*/


FILE *openMatrixFile(char *fileName) {
    /* Reads matrix from mtx file */


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


void populateMatrix(FILE *argFile, float **matrix, float *sum) {
    /* Assign values to the matrix */

    char *line = NULL;
    char *stringNumber;
    size_t len = 0;

    long row, column;
    float value;
    char *ptr;

    printf("\nPopulating matrix...\n");
    while (getline(&line, &len, argFile) != -1) {

        stringNumber = strtok(line, " ");
        column = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        row = strtol(stringNumber, &ptr, 10) - 1;

        stringNumber = strtok(NULL, " ");
        value = strtof(stringNumber, &ptr);

        matrix[row][column] = value;
        sum[row] += value;
    }
    free(line);
}


void normalizeCPM(float **matrix, const float *sum, long numRows, long numColumns) {
    /* Perform CPM normalization */

    printf("\nNormalizing...\n");

    for (int objIdx = 0; objIdx < numRows; objIdx++) {
        if (sum[objIdx] != 0) {
            for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
                matrix[objIdx][columnIdx] = (matrix[objIdx][columnIdx] / sum[objIdx]) * 1e6f;
            }
        }
    }
}


void getJobTimes(struct timespec init, struct timespec reading, struct timespec total) {
    /* Get times spent */

    float readingTime = (reading.tv_sec - init.tv_sec);
    readingTime += (reading.tv_sec - init.tv_sec) / 1000000000.0;
    printf("\nReading Time: %f \n", readingTime);

    float executionTime = (total.tv_sec - init.tv_sec);
    executionTime += (total.tv_nsec - init.tv_nsec) / 1000000000.0;
    printf("Total execution time is: %f\n", executionTime);
}


void writeOutput(float **centers, int *membership, const mtxDimensions job, const float *columnsWeights) {
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
                fprintf(centFiles, "%f ", centers[clusterIdx][centersCol]);
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

/* ____________________________________________________ MinMaxKmeans _______________________________________________*/

void cleanOldKMeans(kMeansOutput oldKMeans) {
    /* Free unused kMeans results */

    free(oldKMeans.cluster);
    free(oldKMeans.clusterWeights);
}


void initializeRandomCenters(const float **matrix, float **centroids, const mtxDimensions job) {
    /* Initialize centroids to random points */

    int randomPoint;

    srand(time(NULL) * job.numObjs);

    printf("\nSetting random centers to");
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        randomPoint = rand() % job.numObjs;
        printf(" %d", randomPoint);

        for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
            centroids[clusterIdx][columnIdx] = matrix[randomPoint][columnIdx];
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


void getEuclideanDist(const float **matrix, float **centroids, float **euclideanDistances, const mtxDimensions job) {
    /* Find euclidean distances for every attribute in every object to each clusters */

    float distance;

    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (int objsIdx = 0; objsIdx < job.numObjs; objsIdx++) {
            distance = 0;

            for (int columnIdx = 0; columnIdx < job.numColumns; columnIdx++) {
                distance += powf((matrix[objsIdx][columnIdx] - centroids[clusterIdx][columnIdx]), 2);
            }

            euclideanDistances[objsIdx][clusterIdx] = sqrtf(distance);
        }
    }
}


void getClusterMeans(const float **matrix, float **centroids, const int *clusterSize, const int *objsMembership,
                     const mtxDimensions job) {
    /* Move centroids to mean between every point in cluster */

    const long numClusters = job.numClusters;
    const long numColumns = job.numColumns;

    float **sumDistances = matrixAlloc(numClusters, numColumns);

    int cluster;

    // Loop over matrix to populate sumDistances
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = objsMembership[objIdx];
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            sumDistances[cluster][columnIdx] += matrix[objIdx][columnIdx];
        }
    }

    // Loop over sumDistances to get means and move centroids
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            centroids[clusterIdx][columnIdx] = sumDistances[clusterIdx][columnIdx] / (float) clusterSize[clusterIdx];
        }
    }

    matrixFree(sumDistances, numClusters);
}


int getNearestCluster(const float *distances, const int numClusters) {
    /* Returns the  nearest cluster from the point */

    int nearestCluster = 0;
    float min_dist = FLT_MAX;

    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        if (min_dist > distances[clusterIdx]) {
            min_dist = distances[clusterIdx];
            nearestCluster = clusterIdx;
        }
    }

    return nearestCluster;
}


void updateClusterWeights(float *clusterWeights, float **objsMtxMembership, float **euclideanDistances,
                          const mtxDimensions job, const float pValue) {
    /* Updates cluster weights according to distances and size */

    const long numClusters = job.numClusters;
    float sumWeightedDistances = 0;
    float *clustersDistance = floatArrayAlloc(numClusters, 1);

    // Loop over clusters to get sum of distances for each obj
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
            clustersDistance[clusterIdx] +=
                    objsMtxMembership[objIdx][clusterIdx] * euclideanDistances[objIdx][clusterIdx];
        }
        sumWeightedDistances += powf(clustersDistance[clusterIdx], (1 / (1 - pValue)));
    }

    // Loop over sum of clusters to update weights
    for (int clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
        clusterWeights[clusterIdx] = powf(clustersDistance[clusterIdx], (1 / (1 - pValue))) / sumWeightedDistances;
    }

    free(clustersDistance);
}


float updateSumWeights(float **weightedDistances, float **euclideanDistances, float *clusterWeights,
                       const float pValue, const mtxDimensions job) {
    /* Get sum of weights to check condition */

    float sumWeights = 0;
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
            weightedDistances[objIdx][clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx][clusterIdx];
            sumWeights += weightedDistances[objIdx][clusterIdx];
        }
    }
    return sumWeights;
}


void copyMatrix(float **dest, float** source, mtxDimensions jobInfo) {
    /* Copy one matrix into another */

    for (int objIdx = 0; objIdx < jobInfo.numObjs; objIdx++) {
        memcpy(dest[objIdx], source[objIdx], sizeof(float) * jobInfo.numClusters);
    }
}


kMeansOutput minMaxKMeans(const float **matrix, float **centroids, const mtxDimensions jobInfo, const int getRandomPoints) {
    /* Main routine for MinMaxKMeans */

    // Initialize variables
    const long numObjs = jobInfo.numObjs, numClusters = jobInfo.numClusters;
    int objIdx, clusterIdx;
    kMeansOutput kMeansResult;
    float pValue = 0;
    const float pStep = 0.01f, pMax = 0.5f;
    int flag = 1;


    // If random is true, initialize centers with random points
    if (getRandomPoints == 1) {
        initializeRandomCenters(matrix, centroids, jobInfo);
    }


    // DELETAR PAR A RANDOMIZAR
    for (int j = 0; j < jobInfo.numColumns; j++) {
        centroids[0][j] = matrix[0][j];
        centroids[1][j] = matrix[2][j];
    }
    printf("Performing MinMaxKMeans...");

    // Initialize matrices and arrays to be used
    float **oldObjsMembership = matrixAlloc(numObjs, numClusters);
    float **weightedDistances = matrixAlloc(numObjs, numClusters);
    float **objsMtxMembership = matrixAlloc(numObjs, numClusters);

    float *clusterWeights = floatArrayAlloc(numClusters, 0);
    float *oldClusterWeights = floatArrayAlloc(numClusters, 0);
    int *objsMembership = intArrayAlloc(numObjs, 0);


    // Initialize cluster weights
    initializeClusterWeights(clusterWeights, oldClusterWeights, numClusters);


    // Initialize euclidean distances
    float **euclideanDistances = matrixAlloc(numObjs, numClusters);
    getEuclideanDist(matrix, centroids, euclideanDistances, jobInfo);


    // Initialize matrix weights
    for (objIdx = 0; objIdx < numObjs; objIdx++) {
        for (clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            weightedDistances[objIdx][clusterIdx] =
                    powf(clusterWeights[clusterIdx], pValue) * euclideanDistances[objIdx][clusterIdx];
        }
    }


    // Variables for loop
    int empty = 0, nIter = 0;
    float oldSumWeights, sumWeights = 0;


    // Loop until weights converge to 0 or hit max loops
    while (1) {
        nIter++;


        // Classify each object and get cluster sizes
        int *clusterSize = intArrayAlloc(numClusters, 1);
        int nearestCluster;
        for (objIdx = 0; objIdx < numObjs; objIdx++) {
            nearestCluster = getNearestCluster(weightedDistances[objIdx], numClusters);
            objsMtxMembership[objIdx][nearestCluster] = 1;
            objsMembership[objIdx] = nearestCluster;
            clusterSize[nearestCluster]++;
        }

        // Lower pValue if some cluster is empty
        for (clusterIdx = 0; clusterIdx < numClusters; clusterIdx++) {
            if (clusterSize[clusterIdx] < 1) {
                empty = 1;
                pValue -= pStep;

                copyMatrix(objsMtxMembership, oldObjsMembership, jobInfo);
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
        getClusterMeans(matrix, centroids, clusterSize, objsMembership, jobInfo);

        // Increase pValue if no cluster is empty
        if ((pValue < pMax) && (empty == 0)) {
            copyMatrix(oldObjsMembership, objsMtxMembership, jobInfo);
            memcpy(oldClusterWeights, clusterWeights, sizeof(float) * numClusters);
            pValue += pStep;
        }

        // Update euclidean distances to new centroids
        getEuclideanDist(matrix, centroids, euclideanDistances, jobInfo);

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


    matrixFree(weightedDistances, numObjs);
    matrixFree(objsMtxMembership, numObjs);
    matrixFree(oldObjsMembership, numObjs);
    matrixFree(euclideanDistances, numObjs);
    free(oldClusterWeights);

    return kMeansResult;
}


kMeansOutput getKMeansResult(const float **matrix, float **centroids, const mtxDimensions jobInfo, int getRandom) {
    /* Perform MinMax KMeans until all clusters are populated */
    int nIter = 0;

    kMeansOutput kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);

    while (kMeansResult.flag == 0 && nIter < 30) {
        getRandom = 1;
        nIter++;
        cleanOldKMeans(kMeansResult);
        kMeansResult = minMaxKMeans(matrix, centroids, jobInfo, getRandom);
    }

    return kMeansResult;
}


/* ____________________________________________________ Update Weights _______________________________________________*/


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




/* ____________________________________________________ Update Centers _______________________________________________*/

void getWeightedObjects(const float **matrix, float **weightedObj, long numObjs, long numColumns, float *weights) {
    /* Calculates weight for every column in every object */

    long numRows = 0;

    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (weights[columnIdx] != 0) {
            for (int objIdx = 0; objIdx < numObjs; objIdx++) {
                weightedObj[objIdx][numRows] = matrix[objIdx][columnIdx] * sqrtf(weights[columnIdx]);
            }
            numRows++;
        }
    }
}


void moveCenterMeanBalanced(float **weightedObjs, float **balancedCenters, const mtxDimensions job,
                            long nonZeroWeights, const int *centroids) {
    /* Moves centers according to calculated weights */

    int cluster, columnIdx;

    int *sumClusterPoints = intArrayAlloc(job.numClusters, 1);

    // Get sum of points in cluster considering weights
    for (int objIdx = 0; objIdx < job.numObjs; objIdx++) {
        cluster = centroids[objIdx];
        sumClusterPoints[cluster]++;
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[cluster][columnIdx] += weightedObjs[objIdx][columnIdx];
        }
    }

    // Calculate mean for each cluster and move center to it
    for (int clusterIdx = 0; clusterIdx < job.numClusters; clusterIdx++) {
        for (columnIdx = 0; columnIdx < nonZeroWeights; columnIdx++) {
            balancedCenters[clusterIdx][columnIdx] =
                    balancedCenters[clusterIdx][columnIdx] / (float) sumClusterPoints[clusterIdx];
        }
    }

    free(sumClusterPoints);
}


int countUniques(const int *uniqueNearest, int numClusters) {
    // Return count of unique values in array

    int ans = 0;

    for (int i = 0; i < numClusters; i++) {
        if (uniqueNearest[i] > 0) {
            ans++;
        }
    }

    return ans;
}


kMeansOutput updateCenters(const float **matrix, mtxDimensions jobInfo, float *columnsWeights,
                           int *membership, int nonZeroWeights) {
    /* Main routine for updating centers according to objects weights */

    printf("Performing centers update...");
    const long numObjs = jobInfo.numObjs, numColumns = jobInfo.numColumns, numClusters = jobInfo.numClusters;

    float **weightedObjs = matrixAlloc(numObjs, nonZeroWeights);
    float **balancedCenters = matrixAlloc(numClusters, nonZeroWeights);

    // Weights every object
    getWeightedObjects(matrix, weightedObjs, numObjs, numColumns, columnsWeights);

    // Move centers according to weights
    moveCenterMeanBalanced(weightedObjs, balancedCenters, jobInfo, nonZeroWeights, membership);

    mtxDimensions updateJob = {numObjs, nonZeroWeights, numClusters};
    // Get euclidean distances
    float **euclideanDistances = matrixAlloc(numObjs, numClusters);
    getEuclideanDist((const float **) weightedObjs, balancedCenters, euclideanDistances, updateJob);

    // Get nearest cluster for each point
    int *clustersMembership = intArrayAlloc(numClusters, 1);
    int nearestCluster;
    for (int objIdx = 0; objIdx < numObjs; objIdx++) {
        nearestCluster = getNearestCluster(euclideanDistances[objIdx], numClusters);
        clustersMembership[nearestCluster]++;
    }
    matrixFree(euclideanDistances, numObjs);

    // Perform MinMaxKMeans with defined centers if every cluster is populated else random

    kMeansOutput kMeansResult;
    if (countUniques(clustersMembership, numClusters) == numClusters) {
        kMeansResult = getKMeansResult((const float **) weightedObjs, balancedCenters, updateJob, 0);
    } else {
        kMeansResult = getKMeansResult((const float **) weightedObjs, balancedCenters, updateJob, 1);
    }

    matrixFree(weightedObjs, numObjs);
    free(clustersMembership);

    printf("Done\n");
    return kMeansResult;
}

/* ____________________________________________________ sparse KMeans _______________________________________________*/

float absoluteSubtract(const float *weights, const float *weights_old, const long numColumns) {
    /* Gets the sum of the difference between two arrays */

    float result = 0;
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        result += fabsf(weights[columnIndex] - weights_old[columnIndex]);
    }

    return result;
}


void initializeWeights(float *columnsWeights, const long numColumns) {
    /* initialize weights array to 1/sqrtf(columns) */

    float initWeight = 1.f / sqrtf((float) numColumns);
    for (int columnIndex = 0; columnIndex < numColumns; columnIndex++) {
        columnsWeights[columnIndex] = initWeight;
    }
}


int getNonZeroWeights(const float *columnWeights, const long numColumns) {
    /* Get number of non zero weight values */

    int nonZero = 0;
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
        if (columnWeights[columnIdx] != 0) nonZero++;
    }

    return nonZero;
}


kMeansOutput sparseKMeans(const float **matrix, const mtxDimensions jobInfo, float *columnsWeights, const int maxLoops) {
    /* Main routine for sparse KMeans */

    // Initialize values
    const long numColumns = jobInfo.numColumns, numObjs = jobInfo.numObjs;
    float boundWeight = 2.f; // should be a list from 1.1 to sqrtf numColumns


    // Initialize columns weights
    float *oldColumnsWeights = floatArrayAlloc(numColumns, 0);
    initializeWeights(columnsWeights, numColumns);


    // Perform MinMaxKMeans once
    int *membershipArray = intArrayAlloc(numObjs, 0);
    float **centroids = matrixAlloc(jobInfo.numClusters, numColumns);

    kMeansOutput kMeansResult = getKMeansResult(matrix, centroids, jobInfo, 1);
    memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);


    // Variables for loop
    int nonZeroWeights = numColumns, nIter = 1;
    float condition = 1;
    const float threshold = 0.0001f;

    // Loop while condition not reached
    while (condition > threshold && nIter < maxLoops) {
        printf("\nRunning loop %d\n", nIter);
        printf("Weights convergence: %f\n ", condition);
        printf("\n");

        memcpy(oldColumnsWeights, columnsWeights, sizeof(float) * numColumns);

        // Updates centers according to cluster weights
        if (nIter > 1) {
            cleanOldKMeans(kMeansResult);
            matrixFree(kMeansResult.centers, jobInfo.numClusters);
            kMeansResult = updateCenters(matrix, jobInfo, columnsWeights, membershipArray, nonZeroWeights);
            memcpy(membershipArray, kMeansResult.cluster, sizeof(int) * numObjs);
        }

        // Updates columns weights
        updatesWeights(matrix, jobInfo, membershipArray, boundWeight, kMeansResult.clusterWeights,
                       kMeansResult.pValue, columnsWeights);
        nonZeroWeights = getNonZeroWeights(columnsWeights, numColumns);

        // Get condition for next loop
        condition = absoluteSubtract(columnsWeights, oldColumnsWeights, numColumns) /
                    absoluteSum(oldColumnsWeights, numColumns);

        nIter++;
    }


    free(oldColumnsWeights);
    free(membershipArray);

    return kMeansResult;
}

/* ____________________________________________________ Main _______________________________________________*/


int main(int argc, char *argv[]) {
    /* Main routine */

    struct timespec initTime, readingTime, totalTime;
    clock_gettime(CLOCK_MONOTONIC, &initTime);


    // Open matrix file
    FILE *mtxFile = openMatrixFile(argv[1]);


    // Reads number of rows and columns from file
    long numObjs, numColumns;
    getMatrixDimensions(mtxFile, &numObjs, &numColumns);


    // Populate matrix and expression sum for normalization
    float *rowSum = floatArrayAlloc(numObjs, 1);
    float **matrix = matrixAlloc(numObjs, numColumns);
    populateMatrix(mtxFile, matrix, rowSum);
    clock_gettime(CLOCK_MONOTONIC, &readingTime);


    // Normalize matrix using CPM
    if (!strcmp(argv[2], "yes")) {
        normalizeCPM(matrix, rowSum, numObjs, numColumns);
    }
    free(rowSum);


    // Initialize variables for KMeans
    const long numClusters = 2;
    const int maxLoops = 10;
    const mtxDimensions jobInfo = {numObjs, numColumns, numClusters};
    float *columnsWeights = floatArrayAlloc(numColumns, 0);



    // Perform Sparse Min Max KMeans
    puts("\nInitializing Sparse KMeans");
    kMeansOutput kMeansResult = sparseKMeans((const float **)matrix, jobInfo, columnsWeights, maxLoops);
    clock_gettime(CLOCK_MONOTONIC, &totalTime);
    puts("\nDone!");


    // Writes output
    writeOutput(kMeansResult.centers, kMeansResult.cluster, jobInfo, columnsWeights);


    // Get job time
    getJobTimes(initTime, readingTime, totalTime);


    matrixFree(matrix, numObjs);
    matrixFree(kMeansResult.centers, numClusters);
    free(kMeansResult.cluster);
    free(kMeansResult.clusterWeights);
    free(columnsWeights);

    return 0;
}
