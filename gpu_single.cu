#include "gpu_single.h"
#include "types.h"

//added grid size 2 dimensions

//1 body functions
__global__
void gpu_one_body_functions_kernel(int* g_s_atomsCnt, atom* g_s_atom_list, query_results* g_s_res) {

    extern __shared__ int sdata[];
    
    long index_x = blockIdx.x * blockDim.x + threadIdx.x;
    long index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    long grid_width = gridDim.x * blockDim.x;
    long i = index_y * grid_width + index_x;

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

    //current atom instance
    atom atomInstance = g_s_atom_list[i];

    atomicAdd(&g_s_res->mass, atomInstance.mass);
    atomicAdd(&g_s_res->charge, atomInstance.charge);

    atomicAdd(&g_s_res->inertiaX, (double)(atomInstance.mass * atomInstance.x));
    atomicAdd(&g_s_res->inertiaY, (double)(atomInstance.mass * atomInstance.y));
    atomicAdd(&g_s_res->inertiaZ, (double)(atomInstance.mass * atomInstance.z));
}

//2 body functions (SDH or POINT DISTANCE HISTOGRAM)
__global__
void gpu_two_body_functions_kernel(atom* at_list, int PDH_acnt, bucket* hist, int num_buckets, double PDH_res) {

    extern __shared__ unsigned long long smem[];

    unsigned long long* shared_histo = smem;
    atom* sharedAtoms = (atom*) &shared_histo[num_buckets];

    long index_x = blockIdx.x * blockDim.x + threadIdx.x;
    long index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    long grid_width = gridDim.x * blockDim.x;
    long index = index_y * grid_width + index_x;

    long i = index;

    //check the bound
    if(i >= PDH_acnt) {
        return;
    }

    //for every first thread of the block
    if(threadIdx.x == 0) {
        for(i = 0; i < num_buckets; i++) {
            shared_histo[i] = 0;
        }

        int start = index;
        int k = 0;
        for(i = start; i < start + blockDim.x && i < PDH_acnt; i++, k++) {
            sharedAtoms[k] = at_list[i];
        }
    }

    __syncthreads();

    i = index;
    
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

    int j;
    for(j = start; j < end; j++) {

        ind2 = j % PDH_acnt;

        double x1 = sharedAtoms[ind1].x;
        double y1 = sharedAtoms[ind1].y;
        double z1 = sharedAtoms[ind1].z;

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

void output_histogram(bucket* hist, int num_buckets){
    int i; 
    unsigned long long total_cnt = 0;
    for(i = 0; i < num_buckets; i++) {
        if(i % 5 == 0) /* we print 5 buckets in a row */
            printf("\n%02d: ", i);
        printf("%15lld ", hist[i].d_cnt);
        total_cnt += hist[i].d_cnt;
        /* we also want to make sure the total distance count is correct */
        if(i == num_buckets - 1)    
            printf("\nT:%lld \n", total_cnt);
        else printf("| ");
    }
}


void run_single_kernel(int atomsCnt, atom* atomList) {

    printf("---------GPU-SINGLE-KERNEL---------\n");

    int BOX_SIZE = 175;
    int PDH_res = 1;

    int num_buckets = BOX_SIZE + 1;

    query_results* res = (query_results*) malloc(sizeof(query_results));
    bucket* histogram = (bucket *)malloc(sizeof(bucket) * num_buckets); 
    
    //set default empty values to remove some garbage inside
    res->mass = 0;
    res->charge = 0;
    res->max_x = 0;
    res->max_y = 0;
    res->max_z = 0;
    res->inertiaX = 0;
    res->inertiaY = 0;
    res->inertiaZ = 0;

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

    
    dim3 block_size;
    //static sizes due to big volume of data
    block_size.x = 1024;
    block_size.y = 1;

    // configure a two dimensional grid as well
    dim3 grid_size;

    int maxGridX = 64000;
    if(atomsCnt < block_size.x * maxGridX) {
        grid_size.x = atomsCnt / block_size.x + 1;
        grid_size.y = 1;
    } else {
        grid_size.x = maxGridX;
        grid_size.y = atomsCnt / (block_size.x * maxGridX);
    }

    printf("grid_size.x/y: %d/%d\n", grid_size.x, grid_size.y);

    //int blockSize = 1024;
    //int gridSize = ceil(atomsCnt / (float)blockSize) + 1;
    //int stripe = 1024 / ;

    /**
    * KERNEL CALLS
    */
    //mass and charge
    //----------------------------------1 BODY KERNEL---------------------------------------------------
    int smem1 = sizeof(float) * block_size.x * 2; //this is not really used for now
    gpu_one_body_functions_kernel<<<grid_size, block_size, smem1, streamComp1 >>>(g_s_atomsCnt, g_s_atom_list, g_s_res);

    //----------------------------------2 BODY KERNEL---------------------------------------------------
    int smem2 = num_buckets * sizeof(unsigned long long) + block_size.x * sizeof(atom);
    gpu_two_body_functions_kernel<<<grid_size, block_size, smem2, streamComp2 >>>(g_s_atom_list, atomsCnt, d_histogram, num_buckets, PDH_res);
    
    cudaStreamSynchronize(streamComp1);
    cudaStreamSynchronize(streamComp2);

    /**
    * DATA COPY TO HOST
    */
    cudaMemcpy(res, g_s_res, sizeof(query_results), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogram, d_histogram, num_buckets * sizeof(bucket), cudaMemcpyDeviceToHost);

    float elapsed = time_calc(start_time); 
    printf("%-40s %.3f\n", "Mass Result: ", res->mass);
    printf("%-40s %.3f\n", "Charge Result: ", res->charge);
    printf("%-40s %.3f\n", "Inertia X Axis: ", res->inertiaX);
    printf("%-40s %.3f\n", "Inertia Y Axis: ", res->inertiaY);
    printf("%-40s %.3f\n", "Inertia Z Axis: ", res->inertiaZ);
    printf("%-40s %.3fmillis\n", "Running time: ", elapsed);
    output_histogram(histogram, num_buckets);


    /**
    * MEM FREE
    */
    cudaFree(g_s_atom_list);
    cudaFree(g_s_res);
    cudaFree(g_s_atomsCnt);
}