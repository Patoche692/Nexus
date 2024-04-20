#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"


struct Mesh
{
	Mesh() = default;
	Mesh(BVH* b, int mId) : bvh(b), materialId(mId) { }

	BVH* bvh;
	int materialId;
};
