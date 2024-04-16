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

void BVHInstance::SetTransform(float3 pos, float3 r, float3 s)
{
	position = pos;
	rotation = r;
	scale = s;
	Mat4 t = Mat4::Translate(pos) * Mat4::RotateX(Utils::ToRadians(r.x))
		* Mat4::RotateY(Utils::ToRadians(r.y)) * Mat4::RotateZ(Utils::ToRadians(r.z)) * Mat4::Scale(s);
	SetTransform(t);
}

void BVHInstance::SetPosition(float3 pos)
{
	SetTransform(pos, rotation, scale);
}

void BVHInstance::SetRotation(float3 axis, float angle)
{
	SetTransform(position, axis * angle, scale);
}

void BVHInstance::SetRotationX(float angle)
{
	SetTransform(position, make_float3(1.0f, 0.0f, 0.0f) * angle, scale);
}

void BVHInstance::SetRotationY(float angle)
{
	SetTransform(position, make_float3(0.0f, 1.0f, 0.0f) * angle, scale);
}

void BVHInstance::SetRotationZ(float angle)
{
	SetTransform(position, make_float3(0.0f, 0.0f, 1.0f) * angle, scale);
}

void BVHInstance::SetScale(float s)
{
	SetTransform(position, rotation, make_float3(s));
}

void BVHInstance::SetScale(float3 s)
{
	SetTransform(position, rotation, s);
}

void BVHInstance::AssignMaterial(int mId)
{
	materialId = mId;
}
