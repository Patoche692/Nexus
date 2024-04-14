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

	void BuildTLAS();

	bool SendDataToDevice();

private:
	std::vector<Mesh> m_Meshes;
	std::set<uint32_t> m_InvalidMeshes;
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<BVH*> m_Blas;
	std::vector<BVHInstance> m_BVHInstances;
	std::set<uint32_t> m_InvalidInstances;
	TLAS m_Tlas;

	Material** m_MaterialSymbolAddress;
	Mesh** m_MeshSymbolAddress;
};
