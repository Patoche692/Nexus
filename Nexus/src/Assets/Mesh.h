#pragma once

#include <vector>
#include "Geometry/BVH/BVH.h"
#include "Geometry/BVH/BVH8Builder.h"
#include "Math/Mat4.h"


struct Mesh
{
	Mesh() = default;
	Mesh(const std::string n, int32_t bId = -1, int32_t mId = -1,
		float3 p = make_float3(0.0f), float3 r = make_float3(0.0f), float3 s = make_float3(1.0f))
		: name(n), materialId(mId), position(p), rotation(r), scale(s), bvhId(bId)
	{
	}
	Mesh(const Mesh& other) = default;
	Mesh(Mesh&& other) = default;

	int32_t bvhId;

	// Transform component of the mesh at loading
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);

	int32_t materialId;
	std::string name;
};
