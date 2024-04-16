#pragma once

#include <iostream>
#include <set>
#include "Geometry/Mesh.h"
#include "Cuda/AssetManager.cuh"
#include "Geometry/Material.h"
#include "Geometry/BVH/BVHInstance.h"
#include "Geometry/BVH/TLAS.h"

class AssetManager
{
public:
	AssetManager();
	~AssetManager();

	void AddMesh(const std::string& filename);

	void AddMaterial();
	void AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();
	std::vector<BVH*> GetBVH() { return m_Bvh; }

	bool SendDataToDevice();

private:
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<BVH*> m_Bvh;

	Material** m_MaterialSymbolAddress;
	Mesh** m_MeshSymbolAddress;
};
