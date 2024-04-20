#pragma once
#include <vector>
#include "BVH/BVHInstance.h"
#include "Mesh.h"

struct MeshInstance
{
	MeshInstance() = default;
	MeshInstance(Mesh& mesh)
	{
		for (BVH* bvh : mesh.bvhs)
		{
			BVHInstance* instance = new BVHInstance(bvh);
			bvhInstances.push_back(instance);
		}
	}

	void SetPosition(float3 p)
	{
		position = p;
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetPosition(position);
	}
	void SetScale(float3 s)
	{
		scale = s;
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetScale(scale);
	}
	void SetScale(float s)
	{
		scale = make_float3(s);
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetScale(scale);
	}
	void SetRotationX(float r)
	{
		rotation = make_float3(r, 0.0f, 0.0f);
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetRotationX(r);
	}
	void SetRotationY(float r)
	{
		rotation = make_float3(0.0f, r, 0.0f);
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetRotationY(r);
	}
	void SetRotationZ(float r)
	{
		rotation = make_float3(0.0f, 0.0f, r);
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->SetRotationZ(r);
	}
	void AssignMaterial(int materialId)
	{
		for (BVHInstance* bvhInstance : bvhInstances)
			bvhInstance->AssignMaterial(materialId);
	}

	std::vector<BVHInstance*> bvhInstances;
	int firstBVHInstanceIdx;
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);
	float3 position = make_float3(0.0f);
};
