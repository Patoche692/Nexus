#include "Scene.h"
#include "Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"
#include "Assets/IMGLoader.h"


Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height, 5.0f, 0.0f))
{
	InitDeviceSceneData();
}

void Scene::Reset()
{
	m_BVHInstances.clear();
	m_InvalidMeshInstances.clear();
	m_MeshInstances.clear();
	m_AssetManager.Reset();
	m_Camera->Invalidate();
}

void Scene::AddMaterial(Material& material)
{
	m_AssetManager.AddMaterial(material);
}

void Scene::BuildTLAS()
{
	m_Tlas = std::make_shared<TLAS>(m_BVHInstances.data(), m_BVHInstances.size());
	m_Tlas->Build();
	newDeviceTLAS(*m_Tlas);
}

MeshInstance& Scene::CreateMeshInstance(uint32_t meshId)
{
	Mesh& mesh = m_AssetManager.GetMeshes()[meshId];

	// Get the raw ptr for BVHInstance because CUDA doesnt support shared_ptr
	m_BVHInstances.push_back(BVHInstance(mesh.bvh8.get()));

	MeshInstance meshInstance(mesh, m_BVHInstances.size() - 1, mesh.materialId);
	m_MeshInstances.push_back(meshInstance);

	InvalidateMeshInstance(m_MeshInstances.size() - 1);

	return m_MeshInstances[m_MeshInstances.size() - 1];
}

void Scene::CreateMeshInstanceFromFile(const std::string& path, const std::string& fileName)
{
	m_AssetManager.AddMesh(path, fileName);
	for (int i = 0; i < m_AssetManager.GetMeshes().size(); i++)
		CreateMeshInstance(i);
	if (m_MeshInstances.size() > 0)
		BuildTLAS();
}

void Scene::AddHDRMap(const std::string& filePath, const std::string& fileName)
{
	m_HdrMap = IMGLoader::LoadIMG(filePath + fileName);
	SendHDRMapToDevice(m_HdrMap);
}

void Scene::InvalidateMeshInstance(uint32_t instanceId)
{
	m_InvalidMeshInstances.insert(instanceId);
}

bool Scene::SendDataToDevice()
{
	bool invalid = false;

	if (m_InvalidMeshInstances.size() != 0)
	{
		for (int i : m_InvalidMeshInstances)
		{
			MeshInstance& meshInstance = m_MeshInstances[i];
			m_BVHInstances[meshInstance.bvhInstanceIdx].SetTransform(meshInstance.position, meshInstance.rotation, meshInstance.scale);
			if (meshInstance.materialId != -1)
				m_BVHInstances[meshInstance.bvhInstanceIdx].AssignMaterial(meshInstance.materialId);
			
		}
		m_Tlas->Build();
		updateDeviceTLAS(*m_Tlas);
		m_InvalidMeshInstances.clear();
		invalid = true;
	}

	if (m_Camera->SendDataToDevice())
		invalid = true;

	if (m_AssetManager.SendDataToDevice())
		invalid = true;

	return invalid;
}
