## Parallel tool for single cell mRNA sequencing (scRNA-seq) analysis   
### Final work developed for the parallel programming discipline    
    
The objective of this work is to develop a tool to cluster the differential expression data, obtained through single cell sequencing, into cells clusters, in a shorter time.    
    
This type of sequencing generates large and sparse matrices, so the Sparse MinMaxKMeans algorithm proposed by Sayak Dey [1] was chosen.    
  
---  
#### Approach  
This repository contains the sequential version and the two approaches developed to parallelize the algorithm.    
  
The first approach used in the parallelization of the algorithm was done with OpenMP.    
    
The second approach used cuda. For this version, it was necessary to use a hybrid model with openMP for cases where the sequencing data is larger than the video card memory, being necessary to process the excess of this data in the CPU.    
  
---  
#### Performance  
For the openMP version the algorithm achieved a speedup of up to 3.6x. The CUDA version achieved speedup of up to 133x, with a sharp drop in performance when the processed matrix size exceeds the GPU memory.    
  
\# Cells | Sequential Time | OpenMP Time | OpenMP Speedup | CUDA+OpenMP Time | CUDA+OpenMP Speedup   
:------: | :-------------: | :---------: | :------------: | :--------------: | :-----------------:  
 15,000 | 1,497.99 | 431.00 | 3.48 | 15.04 | 99.60  
 30,000 | 4,085.45 | 1,181.00 | 3.46 | 31.58 | 129.37  
 60,000| 8,256.37 | 2,473.00 | 3.34 | 61.86 | 133.47  
 **77,035** | 1,699.99 | 472.49 | 3.60 | 104.24 | 16.31  
    
      
The full report with all performance data is also avaliable as `pdf`.   
  
---  
### Compilation  
  
To facilitate compilation, the three generated programs are compiled from a single file.    
  
Use the following command to compile the sequential versions, OpenMP only and OpenMP plus CUDA, respectively:  
        
 gcc -std=c99 -o sequential Sequential/sequential_one_file.c -lm    gcc -std=c99 -fopenmp -o openmp OpenMP/openmp_one_file.c -lm    
    nvcc -Xcompiler " -fopenmp" -o cuda CUDA/cuda_one_file.cu     
   
To run all versions it is necessary to indicate the `mtx` file containing the matrix to be analyzed as the first argument. A matrix of 1,000 is avaliable in the sample folder.     
  
For the sequential version, you must say whether or not you want to perform normalization in CPM, passing ”yes” or ”no” as the second argument. Thus, to run the sequential version without normalization, just use the command:    
  
 ./sequential sample/sample_1000cells_normalised.mtx no  For both OpenMP and CUDA versions you must also specify the number of threads you want. For that, just indicate the number of threads in the second argument, and if you want normalization in the third. For example:    
  
 ./openmp sample/sample_1000cells_normalised.mtx 8 noThis command runs the OpenMP version with 8 threads without normalization.    
  
      
 ./cuda sample/sample_1000cells_normalised.mtx 16 yesAnd this command runs the CUDA version with 16 threads and normalization  
  
---  
#### References  
[1] Dey, S., Das, S., and Mallipeddi, R. (2020). The sparse minmax k-means algorithm for high-dimensional clustering. In Bessiere, C., editor, Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 2103–2110. International Joint Conferences on Artificial Intelligence Organization. Main track.