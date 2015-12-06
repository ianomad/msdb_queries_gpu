#ifndef __INCLUDED_TYPES_H__
#define __INCLUDED_TYPES_H__

typedef struct atom {
	int id;

    float x, y, z; //coordinates
    float vx, vy, vz; //velocities

    float charge;
    float mass;
} atom;

typedef struct query_results {
	
	float mass;
	float charge;
	float max_x;
	float max_y;
	float max_z;
	
} query_results;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

#endif // __INCLUDED_TYPES_H__