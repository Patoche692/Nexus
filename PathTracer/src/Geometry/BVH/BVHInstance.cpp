#include "BVHInstance.h"

void BVHInstance::SetTransform(Mat4& transform)
{
	m_InvTransform = transform.Inverted();
	float3 bMin = m_Bvh->nodes[0].aabbMin;
	float3 bMax = m_Bvh->nodes[0].aabbMax;
	bounds = AABB();
	for (int i = 0; i < 8; i++)
	{
		bounds.Grow(TransformPosition(make_float3(i & 1 ? bMax.x : bMin.x,
			i & 2 ? bMax.y : bMin.y, i & 4 ? bMax.z : bMin.z), transform));
	}
}
