#include "gpu_single.h"
#include "types.h"

//added grid size 2 dimensions

//1 body functions
__global__
void gpu_one_body_functions_kernel(int* g_s_atomsCnt, atom* g_s_atom_list, query_results* g_s_res) {

    extern __shared__ float sdata[];
    
    long index_x = blockIdx.x * blockDim.x + threadIdx.x;
    long index_y = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    long grid_width = gridDim.x * blockDim.x;
    long i = index_y * grid_width + index_x;

    if(i >= *g_s_atomsCnt) {
        return;
    }


    // while(i < *g_s_atomsCnt) {
    //     sdata[tid] += g_s_atom_list[i].mass;
    //     sdata[blockDim.x + tid] += g_s_atom_list[i].charge;

    //     i += blockDim.x;

    //     __syncthreads();
    // }

    float* shared_result = &sdata[0];

    atom atomInstance = g_s_atom_list[i];

    atomicAdd(&shared_result[0], atomInstance.mass);
    atomicAdd(&shared_result[1], atomInstance.charge);
    atomicAdd(&shared_result[2], atomInstance.mass * atomInstance.x);
    atomicAdd(&shared_result[3], atomInstance.mass * atomInstance.y);
    atomicAdd(&shared_result[4], atomInstance.mass * atomInstance.z);
    atomicAdd(&shared_result[5], atomInstance.charge * atomInstance.z);

    __syncthreads();

    if(threadIdx.x == 0) {
        atomicAdd(&g_s_res->mass, shared_result[0]);
        atomicAdd(&g_s_res->charge, shared_result[1]);

        atomicAdd(&g_s_res->inertiaX, shared_result[2]);
        atomicAdd(&g_s_res->inertiaY, shared_result[3]);
        atomicAdd(&g_s_res->inertiaZ, shared_result[4]);

        atomicAdd(&g_s_res->depoleMoment, shared_result[5]);
    }

    // //current atom instance
    // atom atomInstance = g_s_atom_list[i];

    // atomicAdd(&g_s_res->mass, atomInstance.mass);
    // atomicAdd(&g_s_res->charge, atomInstance.charge);

    // atomicAdd(&g_s_res->inertiaX, atomInstance.mass * atomInstance.x);
    // atomicAdd(&g_s_res->inertiaY, atomInstance.mass * atomInstance.y);
    // atomicAdd(&g_s_res->inertiaZ, atomInstance.mass * atomInstance.z);

    // atomicAdd(&g_s_res->depoleMoment, atomInstance.charge * atomInstance.z); //Depole on z
}

//2 body functions (SDH or POINT DISTANCE HISTOGRAM)
__global__
void gpu_two_body_functions_kernel(atom* at_list, int PDH_acnt, bucket* hist, int num_buckets, double bucket_width,
    bool histogram_in_sm) {

    extern __shared__ unsigned long long smem[];

    unsigned long long* shared_histo = smem;
    coordinates* sharedAtoms = histogram_in_sm ? (coordinates*) &shared_histo[num_buckets] : (coordinates*)smem;
    coordinates* sharedAtoms1 = (coordinates*) &sharedAtoms[blockDim.x];

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
        for(i = 0; i < num_buckets && histogram_in_sm; i++) {
            shared_histo[i] = 0;
        }

        int start = index;
        int k = 0;
        for(i = start; i < start + blockDim.x; i++, k++) {
            sharedAtoms[k].x = at_list[i % PDH_acnt].x;
            sharedAtoms[k].y = at_list[i % PDH_acnt].y;
            sharedAtoms[k].z = at_list[i % PDH_acnt].z;

            if(k > 0) {
                sharedAtoms1[k - 1] = sharedAtoms[k];
            }
        }

        // load one more block
        for(; i < start + blockDim.x * 2 + 1; i++, k++) {
            sharedAtoms1[k - 1].x = at_list[i % PDH_acnt].x;
            sharedAtoms1[k - 1].y = at_list[i % PDH_acnt].y;
            sharedAtoms1[k - 1].z = at_list[i % PDH_acnt].z;
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

    int sharedAtoms1Offset = blockDim.x * blockIdx.x + 1;

    int ind1 = threadIdx.x;
    int ind2;

    int k = 0;
    for(ind2 = start; ind2 < end + blockDim.x; ind2 += blockDim.x) {

        double x1 = sharedAtoms[ind1].x;
        double y1 = sharedAtoms[ind1].y;
        double z1 = sharedAtoms[ind1].z;

        double x2, y2, z2;

        int load = 0;
        while(load < blockDim.x && ind2 < end) {
            x2 = sharedAtoms1[ind2 - sharedAtoms1Offset].x;
            y2 = sharedAtoms1[ind2 - sharedAtoms1Offset].y;
            z2 = sharedAtoms1[ind2 - sharedAtoms1Offset].z;

            double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
            int h_pos = (int) (dist / bucket_width);


            if(histogram_in_sm) {
                atomicAdd(&shared_histo[h_pos], 1);
            } else {
                atomicAdd(&hist[h_pos].d_cnt, 1);
            }

            load++;
            ind2++;
        }

        __syncthreads();


        // if(threadIdx.x == 0) { //not finding in shared memory
        //     k = 0;
        //     sharedAtoms1Offset += blockDim.x;
        //     for(i = sharedAtoms1Offset; i < sharedAtoms1Offset + blockDim.x * 2; i++, k++) {
        //         if(i < sharedAtoms1Offset + blockDim.x) {
        //             sharedAtoms1[k].x = sharedAtoms1[k + blockDim.x].x;
        //             sharedAtoms1[k].y = sharedAtoms1[k + blockDim.x].y;
        //             sharedAtoms1[k].z = sharedAtoms1[k + blockDim.x].z;
        //         } else {
        //             sharedAtoms1[k].x = at_list[i % PDH_acnt].x;
        //             sharedAtoms1[k].y = at_list[i % PDH_acnt].y;
        //             sharedAtoms1[k].z = at_list[i % PDH_acnt].z;
        //         }
        //     }
        // }

        __syncthreads();
    }

    __syncthreads();

    if(threadIdx.x == 0 && histogram_in_sm) {
        for(i = 0; i < num_buckets; i++) {
            atomicAdd(&hist[i].d_cnt, shared_histo[i]);
        }
    }
}

void output_histogram(bucket* hist, int num_buckets) {
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


void run_single_kernel(int atomsCnt, atom* atomList, int workload, float bucket_width, float space) {

    int num_buckets = space / bucket_width + 1;

    query_results* res = (query_results*) malloc(sizeof(query_results));
    bucket* histogram = (bucket *)malloc(sizeof(bucket) * num_buckets); 

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

    cudaMemcpy(g_s_atomsCnt, &atomsCnt, sizeof(int), cudaMemcpyHostToDevice);
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

    printf("Grid Sizes: [%d, %d] \n", grid_size.x, grid_size.y);

    //int blockSize = 1024;
    //int gridSize = ceil(atomsCnt / (float)blockSize) + 1;
    //int stripe = 1024 / ;

    int i, w;
    for(w = 0; w < workload; w++) {

        //set default empty values to remove some garbage inside
        res->mass = 0;
        res->charge = 0;
        res->max_x = 0;
        res->max_y = 0;
        res->max_z = 0;
        res->inertiaX = 0;
        res->inertiaY = 0;
        res->inertiaZ = 0;
        res->depoleMoment = 0;

        for(i = 0; i < num_buckets; i++) {
            histogram[i].d_cnt = 0;
        }

        cudaMemcpy(g_s_res, res, sizeof(query_results), cudaMemcpyHostToDevice);
        cudaMemcpy(d_histogram, histogram, num_buckets * sizeof(bucket), cudaMemcpyHostToDevice);

        /**
        * KERNEL CALLS
        */
        //mass and charge
        //----------------------------------1 BODY KERNEL---------------------------------------------------
        int smem1 = sizeof(float) * block_size.x * 10; //this is not really used for now
        gpu_one_body_functions_kernel<<<grid_size, block_size, smem1, streamComp1 >>>(g_s_atomsCnt, g_s_atom_list, g_s_res);

        //----------------------------------2 BODY KERNEL---------------------------------------------------
        int smem2 = num_buckets * sizeof(unsigned long long) + 3 * block_size.x * sizeof(coordinates);
        bool histogram_in_sm = true;
        if(smem2 > 63000) {
            printf("Not able to allocate SM for SDH\n");
            smem2 = 3 * block_size.x * sizeof(coordinates);
            histogram_in_sm = false;
        }

        printf("SMEM size: %d\n", smem2);
        printf("Size of coordinates: %d\n", sizeof(coordinates));
        printf("Size of coordinates array: %d\n", 3 * block_size.x * sizeof(coordinates));
        printf("Size of bucket: %d\n", sizeof(unsigned long long));
        printf("Size of bucket array: %d\n", num_buckets * sizeof(unsigned long long));

        cudaStreamSynchronize(streamComp1);

        gpu_two_body_functions_kernel<<<grid_size, block_size, smem2, streamComp2 >>>(g_s_atom_list, atomsCnt, d_histogram, num_buckets, bucket_width, histogram_in_sm);
        cudaStreamSynchronize(streamComp2);
    }

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
    printf("%-40s %.3f\n", "Depole Moment Z Axis: ", res->depoleMoment);
    printf("%-40s %.3fmillis\n", "Running time: ", elapsed);
    output_histogram(histogram, num_buckets);

    /**
    * MEM FREE
    */
    cudaFree(g_s_atom_list);
    cudaFree(g_s_res);
    cudaFree(g_s_atomsCnt);
    cudaFree(d_histogram);
}