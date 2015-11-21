#include "utils.h"

void shuffle() {
	srand((unsigned)time(NULL));
}

/* 
* Generates random number uniformly distributed (in range of [0-1]) 
*/
double uni_rand() {

	double val = ((double)rand() / (double)RAND_MAX);

	return val;
}

float time_calc(timeval startTime) {
    long sec_diff, usec_diff;
    struct timezone Idunno;
    struct timeval endTime;

    gettimeofday(&endTime, &Idunno);
    
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff = endTime.tv_usec - startTime.tv_usec;

    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }

    return (float) (sec_diff * 1.0 + usec_diff / 1000000.0);
}

void generate_data(atom* atom_list, int atoms_cnt) {
    shuffle();
    
    float space = 1000.0f;

    for(int i = 0; i < atoms_cnt; i++) {
        // atom_list[i].id = i;

        // atom_list[i].x = uni_rand() * space;
        // atom_list[i].y = uni_rand() * space;
        // atom_list[i].z = uni_rand() * space;

        // atom_list[i].charge = uni_rand() * 15;
        atom_list[i].mass = 1;//uni_rand() * 10;
    }
}