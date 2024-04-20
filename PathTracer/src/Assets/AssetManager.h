#pragma once

#include <iostream>
#include <set>
#include "Geometry/Mesh.h"
#include "Cuda/AssetManager.cuh"
#include "Geometry/Material.h"
#include "Geometry/BVH/BVHInstance.h"
#include "Geometry/BVH/TLAS.h"
#include "Texture.h"

class AssetManager
{
public:
	AssetManager();
	~AssetManager();

	void AddMesh(const std::string& filename);

	void AddMaterial();
	int AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();
	std::vector<BVH*> GetBVH() { return m_Bvh; }
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

	int AddTexture(const std::string& filename);
	void ApplyTextureToMaterial(int materialId, int textureId);

	bool SendDataToDevice();

private:
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<Texture> m_Textures;
	std::vector<BVH*> m_Bvh;
	std::vector<Mesh> m_Meshes;

	Material** m_MaterialSymbolAddress;
	Mesh** m_MeshSymbolAddress;
};
