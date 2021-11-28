#ifndef FINAL_STRUCTS_H
#define FINAL_STRUCTS_H

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

#endif //FINAL_STRUCTS_H
