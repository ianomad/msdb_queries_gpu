#include <iostream>

#include "device_functions.h"
#include <cuda_runtime.h>
#include "cuda.h"

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cmath>

#include "types.h"
#include "utils.h"
#include "gpu_single.h"

#include "utils.cu"
#include "gpu_single.cu"


int block_size = 1024; //1025 max (# of threads in a block)

int main(int argc, char *argv[]) {

    int atoms_cnt = 200000;
    int workload = 10;

    atoms_cnt = atoi(argv[1]);
    workload = atoi(argv[2]);

    run_single_kernel(atoms_cnt, workload);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

	return 0;
}
