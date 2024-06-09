#include "BVHInstance.h"
#include "Utils/Utils.h"

void BVHInstance::SetTransform(Mat4& t)
{
	transform = t;
	invTransform = t.Inverted();
	float3 bMin = bvh->nodes[0].p;
	float3 bMax = bMin + make_float3(
		exp2f(bvh->nodes[0].e[0] - 127),
		exp2f(bvh->nodes[0].e[1] - 127),
		exp2f(bvh->nodes[0].e[2] - 127)
	) * (exp2f(8) - 1);
	

	bounds = AABB();
	for (int i = 0; i < 8; i++)
	{
		bounds.Grow(TransformPosition(make_float3(i & 1 ? bMax.x : bMin.x,
			i & 2 ? bMax.y : bMin.y, i & 4 ? bMax.z : bMin.z), t));
	}
}

void BVHInstance::SetTransform(float3 pos, float3 r, float3 s)
{
	Mat4 t = Mat4::Translate(pos) * Mat4::RotateX(Utils::ToRadians(r.x))
		* Mat4::RotateY(Utils::ToRadians(r.y)) * Mat4::RotateZ(Utils::ToRadians(r.z)) * Mat4::Scale(s);
	SetTransform(t);
}

void BVHInstance::AssignMaterial(int mId)
{
	materialId = mId;
}

D_BVHInstance BVHInstance::ToDevice()
{
	D_BVHInstance deviceInstance;
	deviceInstance.bounds = { bounds.bMin, bounds.bMax };

	D_BVH8 deviceBvh = bvh->ToDevice();

	D_BVH8* deviceBvhPtr;
	checkCudaErrors(cudaMalloc((void**)deviceBvhPtr, sizeof(D_BVH8)));
	checkCudaErrors(cudaMemcpy(deviceBvhPtr, &deviceBvh, sizeof(D_BVH8), cudaMemcpyHostToDevice));

	deviceInstance.bvh = deviceBvhPtr;
	deviceInstance.transform = transform;
	deviceInstance.invTransform = invTransform;
	deviceInstance.materialId = materialId;
	return deviceInstance;
}
