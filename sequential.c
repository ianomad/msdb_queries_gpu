#include <iostream>

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

#include <time.h>
#include <stdlib.h>


int block_size = 1024; //1025 max (# of threads in a block)

int main(int argc, char *argv[]) {

    /**
    * Read the number of particles
    */
    int atomsCnt = atoi(argv[1]);

    int BOX_SIZE = 175;
    int PDH_res = 1;

    int num_buckets = BOX_SIZE + 1;

    /**
    * Read name of the file
    */
    std::string fileName = argv[2];
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

    while(!stream.eof()) {
        //read line from file
        std::getline(stream, line);

        std::stringstream lineStream(line);
        
        lineStream >> token;
        if(token.compare("HEAD") == 0) {
            //skip the header

            std::cout << line << std::endl;

            heads++;
            std::cout << "Frame #" << heads << " processing. " << std::endl;
            std::cout << atomCount << " atoms read in previous frame." << std::endl;

            if(atomCount > 0) {
                //here we run sequential
                
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
                res->depoleMoment = 0;

                int i;
                for(i = 0; i < num_buckets; i++) {
                    histogram[i].d_cnt = 0;
                }

                for(i = 0; i < atomsCnt; i++) {
                    res->mass += atomsList[i].mass;
                    res->charge += atomsList[i].charge;
                    res->inertiaX += atomsList[i].mass * atomsList[i].x;
                    res->inertiaY += atomsList[i].mass * atomsList[i].y;
                    res->inertiaZ += atomsList[i].mass * atomsList[i].z;
                    res->depoleMoment += atomsList[i].charge * atomsList[i].z;
                }

            }

            atomCount = 0;
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

        atomsList[atomCount].x = rand() % 100;
        atomsList[atomCount].y = rand() % 100;
        atomsList[atomCount].z = rand() % 100;

        atomCount++;
    }

    printf("\n\n\nHeads: %d\n", heads);
    printf("Atom Count: %d\n", atomCount);
    
    float elapsed = time_calc(start_time);
    printf("%-40s %.3fmillis\n", "Total Running time: ", elapsed);

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
