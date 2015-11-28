#include <iostream>

#include "device_functions.h"
#include <cuda_runtime.h>
#include "cuda.h"
#include <string>
#include <fstream>

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

    /**
    * Read the number of particles
    */
    int numOfParticles = atoi(argv[1]);

    /**
    * Read name of the file
    */
    std::string fileName = argv[2];
    std::ifstream stream(fileName.c_str());
    std::cout << "Reading file: " << fileName << std::endl;

    atom* atoms = (atom*)malloc(sizeof(atom) * numOfParticles);

    int heads = 0;
    int atomCount = 0;

    std::string token;
    while(stream >> token) {
        if(token.compare("HEAD") == 0) {
            heads++;
            atomCount = 0;

            std::cout << "Frame #" << heads << " processing.." << std::endl;
        } else if(token.compare("ATOM")) {
            //example: `ATOM  00000000    00000001    00000001    17.297  15.357  5.428   -0.548  15.9994`

            //frame (just skipping for now)
            stream >> token;
            //number
            stream >> atoms[atomCount].id;
            //type (just skipping for now)
            stream >> token;
            stream >> atoms[atomCount].x;
            stream >> atoms[atomCount].y;
            stream >> atoms[atomCount].z;
            stream >> atoms[atomCount].charge;
            stream >> atoms[atomCount].mass;

            atomCount++;
        }
    }

    printf("Heads: %d\n", heads);
    printf("Atom Count: %d\n", atomCount);


    // int atoms_cnt = 200000;
    // int workload = 10;

    // atoms_cnt = atoi(argv[1]);
    // workload = atoi(argv[2]);

    // run_single_kernel(atoms_cnt, workload);

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
