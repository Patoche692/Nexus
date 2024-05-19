#pragma once

#include <iostream>

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Assets/AssetManager.h"
#include "Geometry/MeshInstance.h"

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);
	void Reset();

	std::shared_ptr<Camera> GetCamera() { return m_Camera; }

	void AddMaterial(Material& material);
	std::vector<Material>& GetMaterials() { return m_AssetManager.GetMaterials(); }
	AssetManager& GetAssetManager() { return m_AssetManager; }
	bool IsEmpty() { return m_MeshInstances.size() == 0; }

	void BuildTLAS();
	MeshInstance& CreateMeshInstance(uint32_t meshId);
	std::vector<MeshInstance>& GetMeshInstances() { return m_MeshInstances; }
	void CreateMeshInstanceFromFile(const std::string& filePath, const std::string& fileName);
	void AddHDRMap(const std::string& filePath, const std::string& fileName);
	void InvalidateMeshInstance(uint32_t instanceId);

	bool SendDataToDevice();

private:
	std::shared_ptr<Camera> m_Camera;

	std::vector<BVHInstance> m_BVHInstances;
	std::vector<MeshInstance> m_MeshInstances;
	std::set<uint32_t> m_InvalidMeshInstances;
	std::shared_ptr<TLAS> m_Tlas;
	Texture m_HdrMap;

	AssetManager m_AssetManager;
};
