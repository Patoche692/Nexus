#include "Scene.h"
#include "Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"
#include "Assets/IMGLoader.h"


Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height, 5.0f, 0.0f))
{
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
	m_Tlas = std::make_shared<TLAS>(m_BVHInstances, m_AssetManager.GetBVHs());
	m_Tlas->Build();
	m_Tlas->UpdateDeviceData();
}

MeshInstance& Scene::CreateMeshInstance(uint32_t meshId)
{
	Mesh& mesh = m_AssetManager.GetMeshes()[meshId];

	m_BVHInstances.push_back(BVHInstance(meshId, &mesh.bvh8));

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
	m_DeviceHdrMap = Texture::ToDevice(m_HdrMap);
}

void Scene::InvalidateMeshInstance(uint32_t instanceId)
{
	m_InvalidMeshInstances.insert(instanceId);
}

D_Scene Scene::ToDevice(Scene& scene)
{
	D_Scene deviceScene;

	bool invalid = false;

	if (scene.m_AssetManager.SendDataToDevice())
		invalid = true;

	DeviceVector<Texture, cudaTextureObject_t>& deviceDiffuseMaps = scene.m_AssetManager.GetDeviceDiffuseMaps();
	DeviceVector<Texture, cudaTextureObject_t>& deviceEmissiveMaps = scene.m_AssetManager.GetDeviceEmissiveMaps();
	DeviceVector<Material, D_Material>& deviceMaterials = scene.m_AssetManager.GetDeviceMaterials();

	deviceScene.diffuseMaps = deviceDiffuseMaps.Data();
	deviceScene.emissiveMaps = deviceEmissiveMaps.Data();
	deviceScene.materials = deviceMaterials.Data();

	deviceScene.hasHdrMap = scene.m_HdrMap.pixels != nullptr;
	// TODO: clear m_DeviceHdrMap when reset
	deviceScene.hdrMap = scene.m_DeviceHdrMap;
	deviceScene.camera = Camera::ToDevice(*scene.m_Camera);
	scene.m_Camera->SetInvalid(false);

	if (scene.m_InvalidMeshInstances.size() != 0)
	{
		for (int i : scene.m_InvalidMeshInstances)
		{
			MeshInstance& meshInstance = scene.m_MeshInstances[i];
			scene.m_BVHInstances[meshInstance.bvhInstanceIdx].SetTransform(meshInstance.position, meshInstance.rotation, meshInstance.scale);
			if (meshInstance.materialId != -1)
				scene.m_BVHInstances[meshInstance.bvhInstanceIdx].AssignMaterial(meshInstance.materialId);
			
		}
		scene.m_Tlas->Build();
		scene.m_Tlas->SetBVHInstances(scene.m_BVHInstances);
		scene.m_Tlas->UpdateDeviceData();

		scene.m_InvalidMeshInstances.clear();
		invalid = true;
	}

	deviceScene.tlas = TLAS::ToDevice(*scene.m_Tlas);

	return deviceScene;
}
