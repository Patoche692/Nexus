#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, std::vector<Triangle>& triangles, int mId)
		: name(n), materialId(mId)
	{
		bvh = std::make_shared<BVH>(triangles);
	}

	std::shared_ptr<BVH> bvh;
	std::string name;
	int materialId;
};
