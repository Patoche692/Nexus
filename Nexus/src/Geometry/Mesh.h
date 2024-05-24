#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"
#include "Geometry/BVH/BVH8Builder.h"
#include "Math/Mat4.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, std::vector<Triangle>& triangles, int mId,
		float3 p = make_float3(0.0f), float3 r = make_float3(0.0f), float3 s = make_float3(1.0f))
		: name(n), materialId(mId), position(p), rotation(r), scale(s)
	{
		bvh = std::make_shared<BVH>(triangles);
		BVH8Builder builder(bvh.get());
		std::cout << n << std::endl;
		builder.Init();
		std::shared_ptr<BVH8> bvh8 = builder.Build();
	}

	std::shared_ptr<BVH> bvh;
	// Transform component of the mesh at loading
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);

	int materialId;
	std::string name;
};
