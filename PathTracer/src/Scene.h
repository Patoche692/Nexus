#pragma once

#include <iostream>

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Assets/AssetManager.h"

enum struct SceneType
{
	CORNELL_BOX,
	CORNELL_BOX_SPHERES,
	DRAGON
};

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);

	std::shared_ptr<Camera> GetCamera() { return m_Camera; }

	void AddMaterial(Material& material);
	std::vector<Material>& GetMaterials() { return m_AssetManager.GetMaterials(); }
	AssetManager& GetAssetManager() { return m_AssetManager; }
	void AddMesh(const std::string& filename) { m_AssetManager.AddMesh(filename); }

	void BuildTLAS();
	MeshInstance& CreateMeshInstance(uint32_t meshId);
	std::vector<MeshInstance>& GetMeshInstances() { return m_MeshInstances; }
	void InvalidateMeshInstance(uint32_t instanceId);

	bool SendDataToDevice();

private:
	std::shared_ptr<Camera> m_Camera;

	std::vector<MeshInstance> m_MeshInstances;
	std::set<uint32_t> m_InvalidInstances;
	TLAS m_Tlas;

	AssetManager m_AssetManager;
};
