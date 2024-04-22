#pragma once
#include <vector>
#include "BVH/BVHInstance.h"
#include "Mesh.h"

struct MeshInstance
{
	MeshInstance() = default;
	MeshInstance(int bvhInstIdx, int mId = -1)
	{
		bvhInstanceIdx = bvhInstIdx;
		materialId = mId;
	}

	void SetPosition(float3 p) { position = p; }
	void SetRotationX(float r) { rotation.x = r; }
	void SetRotationY(float r) { position.y = r; }
	void SetRotationZ(float r) { position.z = r; }
	void SetScale(float s) { scale = make_float3(s); }
	void SetScale(float3 s) { scale = s; }
	void AssignMaterial(int mId) { materialId = mId; }

	int bvhInstanceIdx;
	int materialId = -1;

	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);
	float3 position = make_float3(0.0f);
};
