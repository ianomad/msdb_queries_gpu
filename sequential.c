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
#include "utils.cu"

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

    query_results* res = (query_results*) malloc(sizeof(query_results));
    bucket* histogram = (bucket *)malloc(sizeof(bucket) * num_buckets); 

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

                int i, j;
                for(i = 0; i < num_buckets; i++) {
                    histogram[i].d_cnt = 0;
                }

                //one body
                for(i = 0; i < atomsCnt; i++) {
                    res->mass += atomsList[i].mass;
                    res->charge += atomsList[i].charge;
                    res->inertiaX += atomsList[i].mass * atomsList[i].x;
                    res->inertiaY += atomsList[i].mass * atomsList[i].y;
                    res->inertiaZ += atomsList[i].mass * atomsList[i].z;
                    res->depoleMoment += atomsList[i].charge * atomsList[i].z;
                }

                //two body
                for(i = 0; i < atomsCnt; i++) {
                    float x1 = atomsList[i].x;
                    float y1 = atomsList[i].y;
                    float z1 = atomsList[i].z;

                    for(j = i; j < atomsCnt; j++) {
                        float x2 = atomsList[i].x;
                        float y2 = atomsList[i].y;
                        float z2 = atomsList[i].z;
                        
                        double dist = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
                        int h_pos = (int) (dist / PDH_res);
                        histogram[h_pos]++;
                    }
                }


                float elapsed = time_calc(start_time); 
                printf("%-40s %.3f\n", "Mass Result: ", res->mass);
                printf("%-40s %.3f\n", "Charge Result: ", res->charge);
                printf("%-40s %.3f\n", "Inertia X Axis: ", res->inertiaX);
                printf("%-40s %.3f\n", "Inertia Y Axis: ", res->inertiaY);
                printf("%-40s %.3f\n", "Inertia Z Axis: ", res->inertiaZ);
                printf("%-40s %.3f\n", "Depole Moment Z Axis: ", res->depoleMoment);
                printf("%-40s %.3fmillis\n", "Running time: ", elapsed);
                output_histogram(histogram, num_buckets);
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

    free(res);

	return 0;
}
