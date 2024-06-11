#pragma once

#include <iostream>
#include <set>
#include <thrust/device_vector.h>
#include "Geometry/Mesh.h"
#include "Cuda/AssetManager.cuh"
#include "Geometry/Material.h"
#include "Geometry/BVH/BVHInstance.h"
#include "Geometry/BVH/TLAS.h"
#include "Texture.h"
#include "Cuda/Material.cuh"

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
	std::vector<BVH2*> GetBVH() { return m_Bvh; }
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

	thrust::device_vector<D_Material>& GetDeviceMaterials() { return m_DeviceMaterials; }
	thrust::device_vector<cudaTextureObject_t>& GetDeviceDiffuseMaps() { return m_DeviceDiffuseMaps; }
	thrust::device_vector<cudaTextureObject_t>& GetDeviceEmissiveMaps() { return m_DeviceEmissiveMaps; }

	int AddTexture(const Texture& texture);
	void ApplyTextureToMaterial(int materialId, int diffuseMapId);

	bool SendDataToDevice();

private:
	std::vector<Material> m_Materials;
	std::set<uint32_t> m_InvalidMaterials;
	std::vector<Texture> m_DiffuseMaps;
	std::vector<Texture> m_EmissiveMaps;
	std::vector<BVH2*> m_Bvh;
	std::vector<Mesh> m_Meshes;

	// Device members
	thrust::device_vector<D_Material> m_DeviceMaterials;
	thrust::device_vector<cudaTextureObject_t> m_DeviceDiffuseMaps;
	thrust::device_vector<cudaTextureObject_t> m_DeviceEmissiveMaps;
};
