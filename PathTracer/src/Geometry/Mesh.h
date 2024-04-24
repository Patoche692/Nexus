#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, BVH* b, int mId) : name(n), bvh(b), materialId(mId) { }

	BVH* bvh;
	std::string name;
	int materialId;
};
