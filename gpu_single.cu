#include "gpu_single.h"
#include "types.h"


//1 body functions
__global__
void gpu_one_body_functions_kernel(int* g_s_atomsCnt, atom* g_s_atom_list, query_results* g_s_res) {

    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    
    int i = tid + blockDim.x * blockIdx.x;

    if(i >= *g_s_atomsCnt) {
        return;
    }

    //for some reason shared memory is becoming slower
    // //shared memory structure:

    // //first mass
    // sdata[tid] = g_s_atom_list[i].mass;
    // //second charge
    // sdata[blockDim.x + tid] = g_s_atom_list[i].charge;

    // while(i < *g_s_atomsCnt) {
    //     sdata[tid] += g_s_atom_list[i].mass;
    //     sdata[blockDim.x + tid] += g_s_atom_list[i].charge;

    //     i += blockDim.x;

    //     __syncthreads();
    // }

    // atomicAdd(&g_s_res->mass, sdata[tid]);
    // atomicAdd(&g_s_res->charge, sdata[blockDim.x + tid]);

    atomicAdd(&g_s_res->mass, g_s_atom_list[i].mass);
    atomicAdd(&g_s_res->charge, g_s_atom_list[i].charge);
}

//2 body functions (SDH or POINT DISTANCE HISTOGRAM)
__global__
void gpu_two_body_functions_kernel(atom* at_list, int PDH_acnt, bucket* hist, int num_buckets, double PDH_res) {

    extern __shared__ unsigned long long smem[];

    unsigned long long* shared_histo = smem;
    atom* sharedAtoms = (atom*) &shared_histo[num_buckets];

    int i, j;

    i = blockDim.x * blockIdx.x + threadIdx.x;

    //check the bound
    if(i >= PDH_acnt) {
        return;
    }

    //for every first thread of the block
    if(threadIdx.x == 0) {
        for(i = 0; i < num_buckets; i++) {
            shared_histo[i] = 0;
        }

        int start = blockDim.x * blockIdx.x + threadIdx.x;
        int k = 0;
        for(i = start; i < start + blockDim.x && i < PDH_acnt; i++, k++) {
            sharedAtoms[k] = at_list[i];
        }
    }

    __syncthreads();

    i = blockDim.x * blockIdx.x + threadIdx.x;
    
    int threadLoad = (PDH_acnt + 1) / 2;

    int start = i + 1;
    int end = i + threadLoad;

    if(PDH_acnt % 2 == 0 && i < PDH_acnt / 2) {
        end++;
    }

    int bi = blockDim.x * blockIdx.x;   // block start
    int ei = bi + blockDim.x;           // block end

    int ind1 = threadIdx.x;             // in this block from sharedAtoms
    int ind2;

    for(j = start; j < end; j++) {

        ind2 = j % PDH_acnt;

        double x1 = sharedAtoms[ind1].x_pos;
        double y1 = sharedAtoms[ind1].y_pos;
        double z1 = sharedAtoms[ind1].z_pos;

        double x2, y2, z2;
        
        if(bi <= ind2 && ind2 < ei) {
            x2 = sharedAtoms[ind2 - bi].x;
            y2 = sharedAtoms[ind2 - bi].y;
            z2 = sharedAtoms[ind2 - bi].z;
        } else {
            x2 = at_list[ind2].x;
            y2 = at_list[ind2].y;
            z2 = at_list[ind2].z;
        }

        double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
        int h_pos = (int) (dist / PDH_res);
        atomicAdd(&shared_histo[h_pos], 1);
    }

    __syncthreads();

    if(threadIdx.x == 0) {
        for(i = 0; i < num_buckets; i++) {
            atomicAdd(&hist[i].d_cnt, shared_histo[i]);
        }
    }
}




void run_single_kernel(int atomsCnt, atom* atomList) {

    printf("---------GPU-SINGLE-KERNEL---------\n");

    int BOX_SIZE = 30;
    int PDH_res = 500;

    int num_buckets = BOX_SIZE + 1;

    query_results* res = (query_results*) malloc(sizeof(query_results));
    bucket* histogram = (bucket *)malloc(sizeof(bucket) * num_buckets); 
    
    res->mass = 0;
    res->charge = 0;
    res->max_x = 0;
    res->max_y = 0;
    res->max_z = 0;

    int i;
    for(i = 0; i < num_buckets; i++) {
        histogram[i].d_cnt = 0;
    }

    struct timezone i_dunno;
    struct timeval start_time;
    cudaStream_t streamComp1, streamComp2;

    //Device Types
    int* g_s_atomsCnt;
    atom* g_s_atom_list;
    bucket* d_histogram;
    query_results* g_s_res;

    gettimeofday(&start_time, &i_dunno);

    cudaStreamCreate(&streamComp1);
    cudaStreamCreate(&streamComp2);

    cudaMalloc((void**)&g_s_res, sizeof(query_results));
    cudaMalloc((void**)&g_s_atom_list, sizeof(atom) * atomsCnt);
    cudaMalloc((void**)&g_s_atomsCnt, sizeof(int));
    cudaMalloc((void**)&d_histogram, num_buckets * sizeof(bucket));

    cudaMemcpy(g_s_res, res, sizeof(query_results), cudaMemcpyHostToDevice);
    cudaMemcpy(g_s_atomsCnt, &atomsCnt, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, histogram, num_buckets * sizeof(bucket), cudaMemcpyHostToDevice);
    cudaMemcpy(g_s_atom_list, atomList, sizeof(atom) * atomsCnt, cudaMemcpyHostToDevice);

    /**
    * KERNEL CALL
    */
    int blockSize = 512;
    int gridSize = ceil(atomsCnt / (float)blockSize) + 1;
    //int stripe = 1024 / ;

    //mass and charge
    //----------------------------------1 BODY KERNEL---------------------------------------------------
    int smem1 = sizeof(float) * blockSize * 2;
    gpu_one_body_functions_kernel<<<1, gridSize, smem1, streamComp1 >>>(g_s_atomsCnt, g_s_atom_list, g_s_res);

    //----------------------------------2 BODY KERNEL---------------------------------------------------
    int smem2 = num_buckets * sizeof(unsigned long long) + blockSize * sizeof(atom);
    gpu_two_body_functions_kernel<<<1, gridSize, smem2, streamComp2 >>>(g_s_atom_list, atomsCnt, d_histogram, num_buckets, PDH_res);
    
    cudaStreamSynchronize(streamComp1);
    cudaStreamSynchronize(streamComp2);

    /**
    * DATA COPY TO HOST
    */
    cudaMemcpy(res, g_s_res, sizeof(query_results), cudaMemcpyDeviceToHost);

    float elapsed = time_calc(start_time); 
    printf("%-40s %.3f\n", "Mass Result: ", res->mass);
    printf("%-40s %.3f\n", "Charge Result: ", res->charge);
    printf("%-40s %.3fmillis\n", "Running time: ", elapsed);


    /**
    * MEM FREE
    */
    cudaFree(g_s_atom_list);
    cudaFree(g_s_res);
    cudaFree(g_s_atomsCnt);
}