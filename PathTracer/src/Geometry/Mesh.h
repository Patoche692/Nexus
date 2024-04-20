#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"


struct Mesh
{
	Mesh() = default;
	Mesh(std::vector<BVH*> b) : bvhs(b) { }

	std::vector<BVH*> bvhs;
};
