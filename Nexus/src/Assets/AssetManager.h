#pragma once

#include <iostream>
#include <set>
#include "Memory/Device/DeviceVector.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Geometry/BVH/BVHInstance.h"
#include "Geometry/BVH/TLAS.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"

class AssetManager
{
public:
	AssetManager() = default;

	void Reset();

	void AddMesh(const std::string& path, const std::string filename);

	void AddMaterial();
	int AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();
	std::vector<BVH8> GetBVHs() { return m_Bvhs; }
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

	DeviceVector<Material, D_Material>& GetDeviceMaterials() { return m_DeviceMaterials; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceDiffuseMaps() { return m_DeviceDiffuseMaps; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceEmissiveMaps() { return m_DeviceEmissiveMaps; }

	int AddTexture(const Texture& texture);
	void ApplyTextureToMaterial(int materialId, int diffuseMapId);

	bool SendDataToDevice();

private:
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<Texture> m_DiffuseMaps;
	std::vector<Texture> m_EmissiveMaps;
	std::vector<BVH8> m_Bvhs;
	std::vector<Mesh> m_Meshes;

	// Device members
	DeviceVector<Material, D_Material> m_DeviceMaterials;
	DeviceVector<Texture, cudaTextureObject_t> m_DeviceDiffuseMaps;
	DeviceVector<Texture, cudaTextureObject_t> m_DeviceEmissiveMaps;
};
