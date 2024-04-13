#pragma once

#include <iostream>

#include "Camera.h"
#include "Geometry/Sphere.h"
#include "Assets/AssetManager.h"

class Scene
{
public:
	Scene(uint32_t width, uint32_t height);

	bool IsInvalid() const { return m_Invalid; }
	void Invalidate() { m_Invalid = true; }

	std::shared_ptr<Camera> GetCamera() { return m_Camera; }

	void AddMaterial(Material& material);
	std::vector<Material>& GetMaterials() { return m_AssetManager.GetMaterials(); }
	AssetManager& GetAssetManager() { return m_AssetManager; }
	void AddMesh(const std::string& filename) { m_AssetManager.AddMesh(filename); }

	void SendDataToDevice();

private:
	bool m_Invalid = true;
	std::shared_ptr<Camera> m_Camera;

	AssetManager m_AssetManager;
};
