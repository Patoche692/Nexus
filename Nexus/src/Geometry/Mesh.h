#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"
#include "Math/Mat4.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, std::vector<Triangle>& triangles, int mId)
		: name(n), materialId(mId)
	{
		bvh = std::make_shared<BVH>(triangles);
	}

	std::shared_ptr<BVH> bvh;
	// Projection matrix of the mesh at loading
	Mat4 projection;

	int materialId;
	std::string name;
};
