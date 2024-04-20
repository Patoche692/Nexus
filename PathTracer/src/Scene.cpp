#include "Scene.h"
#include "Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height, 5.0f, 0.0f))
{
}

void Scene::AddMaterial(Material& material)
{
	m_AssetManager.AddMaterial(material);
}

void Scene::BuildTLAS()
{
	m_Tlas = TLAS(m_BVHInstances.data(), m_BVHInstances.size());
	m_Tlas.Build();
	newDeviceTLAS(m_Tlas);
}

MeshInstance& Scene::CreateMeshInstance(uint32_t meshId)
{
	MeshInstance instance(m_AssetManager.GetMeshes()[meshId]);

	instance.firstBVHInstanceIdx = m_BVHInstances.size();

	for (BVHInstance& bvhInstance : instance.bvhInstances)
		m_BVHInstances.push_back(std::move(bvhInstance));

	m_MeshInstances.push_back(std::move(instance));

	return m_MeshInstances[m_MeshInstances.size() - 1];
}

void Scene::InvalidateMeshInstance(uint32_t instanceId)
{
	for (int i = 0; i < m_MeshInstances[instanceId].bvhInstances.size(); i++)
		m_InvalidInstances.insert(m_MeshInstances[instanceId].firstBVHInstanceIdx + i);
}

bool Scene::SendDataToDevice()
{
	bool invalid = false;

	if (m_InvalidInstances.size() != 0)
	{
		for (int i : m_InvalidInstances)
		{
			BVHInstance& instance = m_BVHInstances[i];
			instance.SetTransform(instance.position, instance.rotation, instance.scale);
		}
		m_Tlas.Build();
		updateDeviceTLAS(m_Tlas);
		m_InvalidInstances.clear();
		invalid = true;
	}

	if (m_Camera->SendDataToDevice())
		invalid = true;

	if (m_AssetManager.SendDataToDevice())
		invalid = true;

	return invalid;
}
