#include "BVHInstance.h"
#include "Utils/Utils.h"

void BVHInstance::SetTransform(Mat4& t)
{
	transform = t;
	invTransform = t.Inverted();
	float3 bMin = bvh->nodes[0].aabbMin;
	float3 bMax = bvh->nodes[0].aabbMax;
	bounds = AABB();
	for (int i = 0; i < 8; i++)
	{
		bounds.Grow(TransformPosition(make_float3(i & 1 ? bMax.x : bMin.x,
			i & 2 ? bMax.y : bMin.y, i & 4 ? bMax.z : bMin.z), t));
	}
}

void BVHInstance::Translate(float3 pos)
{
	Mat4 t = Mat4::Translate(pos) * transform;
	SetTransform(t);
}

void BVHInstance::Rotate(float3 axis, float angle)
{
	Mat4 t = Mat4::Rotate(axis, angle * M_PI / 180.0f) * transform;
	SetTransform(t);
}

void BVHInstance::RotateX(float angle)
{
	Mat4 t = Mat4::RotateX(angle * M_PI / 180.0f) * transform;
	SetTransform(t);
}

void BVHInstance::RotateY(float angle)
{
	Mat4 t = Mat4::RotateY(angle * M_PI / 180.0f) * transform;
	SetTransform(t);
}

void BVHInstance::RotateZ(float angle)
{
	Mat4 t = Mat4::RotateZ(angle * M_PI / 180.0f) * transform;
	SetTransform(t);
}

void BVHInstance::Scale(float scale)
{
	Mat4 t = Mat4::Scale(scale) * transform;
	SetTransform(t);
}

void BVHInstance::Scale(float3 scale)
{
	Mat4 t = Mat4::Scale(scale) * transform;
	SetTransform(t);
}

void BVHInstance::AssignMaterial(int mId)
{
	materialId = mId;
}
