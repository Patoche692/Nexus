#pragma once

#include <iostream>

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Geometry/Materials/MaterialManager.h"
#include "Assets/AssetManager.h"

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);

	bool IsInvalid() const { return m_Invalid; }
	void Invalidate() { m_Invalid = true; }

	std::vector<Sphere>& GetSpheres() { return m_Spheres; }
	std::shared_ptr<Camera> GetCamera() { return m_Camera; }

	void AddSphere();
	void AddSphere(Sphere sphere);
	void AddSphere(int materialId);
	void AddMaterial(Material* material);
	std::vector<Material>& GetMaterials() { return m_MaterialManager.GetMaterials(); }
	MaterialManager& GetMaterialManager() { return m_MaterialManager; }
	AssetManager& GetAssetManager() { return m_AssetManager; }
	void AddMesh(const std::string& filename) { m_AssetManager.AddMesh(filename); }

	void SendDataToDevice();

private:
	bool m_Invalid = true;
	std::shared_ptr<Camera> m_Camera;
	std::vector<Sphere> m_Spheres;

	MaterialManager m_MaterialManager;
	AssetManager m_AssetManager;
};
