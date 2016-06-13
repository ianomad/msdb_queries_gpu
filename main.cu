#include <iostream>

#include "device_functions.h"
#include <cuda_runtime.h>
#include "cuda.h"
#include <string>
#include <fstream>
#include <sstream>

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
#include <time.h>
#include <stdlib.h>


int block_size = 1024; //1025 max (# of threads in a block)

int main(int argc, char *argv[]) {

    /**
    * Read the number of particles
    */
    int atomsCnt = atoi(argv[1]);

    /**
    * Read name of the file
    */
    std::string fileName = argv[2];

    int workload = 1;
    if(argc > 3) {
        workload = atoi(argv[3]);
    }

    float bucket_width, space;
    if(argc < 6) {
        printf("Usage: ./main atomCount file workload bucket-width space-width\n");
        exit(1);
    }

    bucket_width = atof(argv[4]);
    space = atof(argv[5]);

    std::ifstream stream(fileName.c_str());
    std::cout << "Reading file: " << fileName << std::endl;

    atom* atomsList = new atom[atomsCnt];

    int heads = 0;
    int atomCount = 0;

    //seed the random generator
    srand(time(NULL));

    std::string token;
    std::string line;

    struct timezone i_dunno;
    struct timeval start_time;

    gettimeofday(&start_time, &i_dunno);

    while(std::getline(stream, line)) {
        //read line from file
        std::stringstream lineStream(line);
        
        lineStream >> token;

        if(token.compare("HEAD") == 0) {
            if(atomCount > 0) {
                std::cout << "**********************Frame #" << heads << "*****************" << std::endl;
                std::cout << atomCount << " atoms read." << std::endl;
                run_single_kernel(atomsCnt, atomsList, workload, bucket_width, space);
            }
            
            // check for error
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }

            atomCount = 0;
            heads++;
            continue;
        }

        //example: `ATOM  00000000    00000001    00000001    17.297  15.357  5.428   -0.548  15.9994`
        //skip some stuff
        lineStream >> token;

        //std::cout << token << std::endl;

        lineStream >> token;
        lineStream >> token;

        //double x, y, z, charge, mass;
        lineStream >> atomsList[atomCount].x;
        lineStream >> atomsList[atomCount].y;
        lineStream >> atomsList[atomCount].z;
        lineStream >> atomsList[atomCount].charge;
        lineStream >> atomsList[atomCount].mass;

        atomsList[atomCount].x = ((rand() % 100) / 100.0f) * space;
        atomsList[atomCount].y = ((rand() % 100) / 100.0f) * space;
        atomsList[atomCount].z = ((rand() % 100) / 100.0f) * space;
        
        atomCount++;
    }

    //printf("\n\n\nHeads: %d\n", heads);
    //printf("Atom Count: %d\n", atomCount);
    
    float elapsed = time_calc(start_time);
    printf("%-40s %.3fmillis\n", "Total Running time: ", elapsed);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

	return 0;
}
