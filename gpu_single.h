#ifndef __INCLUDED_GPU_SINGLE__H__
#define __INCLUDED_GPU_SINGLE__H__

#include "types.h"

__global__ void gpu_one_body_functions_kernel(int* g_s_atoms_cnt, atom* g_s_atom_list, query_results* g_s_res);
__global__ void gpu_two_body_functions_kernel(atom* at_list, int PDH_acnt, bucket* hist, int num_buckets, double PDH_res);

void run_single_kernel(int atomsCnt, atom* atomList);

#endif