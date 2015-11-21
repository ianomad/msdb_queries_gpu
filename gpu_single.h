#ifndef __INCLUDED_GPU_SINGLE__H__
#define __INCLUDED_GPU_SINGLE__H__

#include "types.h"

__global__ void gpu_single_kernel(int* g_s_atoms_cnt, atom* g_s_atom_list, query_results* g_s_res);

void run_single_kernel(int atoms_cnt, int workload);

#endif