
#ifndef FINAL_PARALLEL_STRUCT_H
#define FINAL_PARALLEL_STRUCT_H

// Struct to hold kMeans results
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
#endif //FINAL_PARALLEL_STRUCT_H
