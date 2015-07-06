#ifndef __INCLUDED_TYPES_H__
#define __INCLUDED_TYPES_H__

typedef struct atom {
	int id;

    float x, y, z; //coordinates
    float vx, vy, vz; //velocities

    float charge;
    float mass;
} atom;

#endif // __INCLUDED_TYPES_H__