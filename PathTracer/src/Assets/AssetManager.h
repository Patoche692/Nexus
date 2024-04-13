#pragma once

#include <iostream>
#include <set>
#include "Geometry/Mesh.h"
#include "Cuda/AssetManager.cuh"
#include "Geometry/Material.h"

class AssetManager
{
public:
	AssetManager();
	~AssetManager();

	void AddMesh(const std::string& filename, int materialId = -1);
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }
	void InvalidateMesh(uint32_t index);
	void InvalidateMeshes();

	void AddMaterial();
	void AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();

	bool SendDataToDevice();

private:
	std::vector<Mesh> m_Meshes;
	std::set<uint32_t> m_InvalidMeshes;
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;

	Material** m_MaterialSymbolAddress;
	Mesh** m_MeshSymbolAddress;
};
