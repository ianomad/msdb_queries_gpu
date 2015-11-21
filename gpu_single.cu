#include "gpu_single.h"
#include "types.h"

__global__ void gpu_single_kernel(int* g_s_atoms_cnt, atom* g_s_atom_list, query_results* g_s_res) {

    extern __shared__ int sdata[];

    int tid = threadIdx.x;

    if(tid >= *g_s_atoms_cnt) {
        return;
    }
    
    int i = tid + blockDim.x;
    sdata[tid] = g_s_atom_list[tid].mass;

    while(i < *g_s_atoms_cnt) {
        sdata[tid] += g_s_atom_list[i].mass;
        i += blockDim.x;
        __syncthreads();
    }

    atomicAdd(&g_s_res->mass, sdata[tid]);
}

void run_single_kernel(int atoms_cnt, int workload) {

    printf("---------GPU-SINGLE-KERNEL---------\n");
    atom* atom_list = (atom*)malloc(sizeof(atom) * atoms_cnt);

    query_results* res = (query_results*) malloc(sizeof(query_results));
    res->mass = 0;

    int w;
    int block_size = 1024;
    struct timezone i_dunno;
    struct timeval start_time;
    cudaStream_t streamComp;
    cudaStream_t streamCpy;

    cudaEvent_t start, stop;
    float elapsedTime;

    int* g_s_atoms_cnt;
    atom* g_s_atom_list;
    atom* g_s_atom_list_pinned;
    query_results* g_s_res;

    gettimeofday(&start_time, &i_dunno);

    cudaStreamCreate(&streamComp);
    cudaStreamCreate(&streamCpy);

    cudaMalloc((void**)&g_s_res, sizeof(query_results));
    cudaMalloc((void**)&g_s_atom_list, sizeof(atom) * atoms_cnt);
    cudaMalloc((void**)&g_s_atom_list_pinned, sizeof(atom) * atoms_cnt);
    cudaMalloc((void**)&g_s_atoms_cnt, sizeof(int));

    cudaMemcpy(g_s_res, res, sizeof(query_results), cudaMemcpyHostToDevice);
    cudaMemcpy(g_s_atoms_cnt, &atoms_cnt, sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    generate_data(atom_list, atoms_cnt);

    for(w = 0; w < workload; w++) {
        
        cudaMemcpyAsync(g_s_atom_list, atom_list, sizeof(atom) * atoms_cnt, cudaMemcpyHostToDevice, streamCpy);
        
        cudaStreamSynchronize(streamCpy);
        cudaStreamSynchronize(streamComp);

        /**
        * KERNEL CALL
        */
        int grid_size = ceil(atoms_cnt / (float)block_size) + 1;// + (atoms_cnt % block_size == 0 ? 0 : 1);
        int stripe = 1024;
        gpu_single_kernel<<<1, stripe, sizeof(float) * stripe, streamComp >>>(g_s_atoms_cnt, g_s_atom_list, g_s_res);

        atom* tmp = g_s_atom_list;
        g_s_atom_list = g_s_atom_list_pinned;
        g_s_atom_list_pinned = tmp;
    }
    
    cudaStreamSynchronize(streamComp);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    /**
    * DATA COPY TO HOST
    */
    cudaMemcpy(res, g_s_res, sizeof(query_results), cudaMemcpyDeviceToHost);

    printf("%-40s %.2f\n", "Sum of masses:", res->mass);
    //float elapsed = time_calc(start_time); 
    printf("%-40s %.3fsec\n", "Running time: ", elapsedTime / 1000.0f);

    /**
    * MEM FREE
    */
    cudaFree(g_s_atom_list);
    cudaFree(g_s_res);
    cudaFree(g_s_atoms_cnt);
}