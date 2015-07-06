#include "utils.h"

/*
	utils.h implementation
*/

void shuffle() {
	srand((unsigned)time(NULL));
}

/* 	Generates random number uniformly distributed (in range of [0-1]) */
double uni_rand() {
	double val = ((double)rand() / (double)RAND_MAX);
	return val;
}
