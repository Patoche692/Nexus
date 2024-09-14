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
		std::cout << "Mesh: " << n << std::endl;
		std::cout << "Triangle count: " << triangles.size() << std::endl;

		BVH8Builder builder(triangles);
		builder.Init();
		bvh8 = builder.Build();
		bvh8.InitDeviceData();
	}
	Mesh(const Mesh& other) = default;
	Mesh(Mesh&& other) = default;

	BVH8 bvh8;

	// Transform component of the mesh at loading
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);

	int materialId;
	std::string name;
};
