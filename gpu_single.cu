#include "gpu_single.h"
#include "types.h"


//1 body functions
__global__
void gpu_single_kernel(int* g_s_atomsCnt, atom* g_s_atom_list, query_results* g_s_res) {

    extern __shared__ int sdata[];

    int tid = threadIdx.x;

    if(tid >= *g_s_atomsCnt) {
        return;
    }
    
    int i = tid + blockDim.x;
    sdata[tid] = g_s_atom_list[tid].mass;

    while(i < *g_s_atomsCnt) {
        sdata[tid] += g_s_atom_list[i].mass;
        i += blockDim.x;
        __syncthreads();
    }

    atomicAdd(&g_s_res->mass, sdata[tid]);
}

void run_single_kernel(int atomsCnt, atom* atomList) {

    printf("---------GPU-SINGLE-KERNEL---------\n");
    query_results* res = (query_results*) malloc(sizeof(query_results));
    res->mass = 0;
    res->charge = 0;
    res->max_x = 0;
    res->max_y = 0;
    res->max_z = 0;

    struct timezone i_dunno;
    struct timeval start_time;
    cudaStream_t streamComp;

    int* g_s_atomsCnt;
    atom* g_s_atom_list;
    query_results* g_s_res;

    gettimeofday(&start_time, &i_dunno);

    cudaStreamCreate(&streamComp);

    cudaMalloc((void**)&g_s_res, sizeof(query_results));
    cudaMalloc((void**)&g_s_atom_list, sizeof(atom) * atomsCnt);
    cudaMalloc((void**)&g_s_atomsCnt, sizeof(int));

    cudaMemcpy(g_s_res, res, sizeof(query_results), cudaMemcpyHostToDevice);
    cudaMemcpy(g_s_atomsCnt, &atomsCnt, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(g_s_atom_list, atomList, sizeof(atom) * atomsCnt, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(streamComp);

    /**
    * KERNEL CALL
    */
    int blockSize = 1024;
    int gridSize = ceil(atomsCnt / (float)blockSize) + 1;
    //int stripe = 1024 / ;

    int sizeOfSharedMem = sizeof(float) * gridSize;
    gpu_single_kernel<<<1, gridSize, sizeOfSharedMem, streamComp >>>(g_s_atomsCnt, g_s_atom_list, g_s_res);
    
    cudaStreamSynchronize(streamComp);

    /**
    * DATA COPY TO HOST
    */
    cudaMemcpy(res, g_s_res, sizeof(query_results), cudaMemcpyDeviceToHost);

    float elapsed = time_calc(start_time); 
    printf("%-40s %.3fsec\n", "Running time: ", elapsed / 1000.0f);

    /**
    * MEM FREE
    */
    cudaFree(g_s_atom_list);
    cudaFree(g_s_res);
    cudaFree(g_s_atomsCnt);
}