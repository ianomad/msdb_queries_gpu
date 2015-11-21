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

#endif // __INCLUDED_TYPES_H__