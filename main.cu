#include <iostream>

#include <vector>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "types.h"
#include "utils.cu"

int atoms_cnt = 10000;
float block_size = 1000.0f;

int cuda_threads_cnt = 256;
int cuda_blockss_cnt = 256;

__global__ void gpu_kernel_run() {

    int i;

    i = blockDim.x + blockIdx.x + threadIdx.x;

    // if(i >= atoms_cnt) {
    //     return;
    // }
}

int main() {

    shuffle();

    atom* atom_list = (atom*)malloc(sizeof(atom) * atoms_cnt);

    for(int i = 0; i < atoms_cnt; i++) {
        atom_list[i].id = i;

        atom_list[i].x = uni_rand() * block_size;
        atom_list[i].y = uni_rand() * block_size;
        atom_list[i].z = uni_rand() * block_size;

        atom_list[i].charge = uni_rand() * 15;
        atom_list[i].mass = uni_rand() * 100;
        //printf("x: %f, y: %f, z: %f\n", atom_list[i].x, atom_list[i].y, atom_list[i].z);
    }

    atom* d_atom_list;

    cudaMalloc((void**)&d_atom_list, sizeof(atom) * atoms_cnt);
    cudaMemcpy(d_atom_list, atom_list, sizeof(atom) * atoms_cnt, cudaMemcpyHostToDevice);

    gpu_kernel_run<<<atoms_cnt,1>>>();

    cudaDeviceSynchronize();

    cudaFree(d_atom_list);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    else 
    {
        printf("Success!\n");        
    }

	return 0;
}
