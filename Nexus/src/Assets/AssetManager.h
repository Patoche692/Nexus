#pragma once

#include <iostream>
#include <set>
#include "Device/DeviceVector.h"
#include "Assets/Mesh.h"
#include "Assets/Material.h"
#include "Geometry/BVH/BVHInstance.h"
#include "Geometry/BVH/TLAS.h"
#include "Texture.h"
#include "Cuda/Scene/Material.cuh"

class AssetManager
{
public:
	AssetManager();

	void Reset();

	int32_t CreateBVH(const std::vector<Triangle>& triangles);
	int32_t AddMesh(Mesh&& mesh);

	void InitDeviceData();

	void AddMaterial();
	int AddMaterial(const Material& material);
	std::vector<Material>& GetMaterials() { return m_Materials; }
	void InvalidateMaterial(uint32_t index);
	std::string GetMaterialTypesString();
	std::string GetMaterialsString();
	std::vector<BVH8>& GetBVHs() { return m_Bvhs; }
	std::vector<Mesh>& GetMeshes() { return m_Meshes; }

	DeviceVector<Material, D_Material>& GetDeviceMaterials() { return m_DeviceMaterials; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceDiffuseMaps() { return m_DeviceDiffuseMaps; }
	DeviceVector<Texture, cudaTextureObject_t>& GetDeviceEmissiveMaps() { return m_DeviceEmissiveMaps; }

	const DeviceVector<Material, D_Material>& GetDeviceMaterials() const { return m_DeviceMaterials; }
	const DeviceVector<Texture, cudaTextureObject_t>& GetDeviceDiffuseMaps() const { return m_DeviceDiffuseMaps; }
	const DeviceVector<Texture, cudaTextureObject_t>& GetDeviceEmissiveMaps() const { return m_DeviceEmissiveMaps; }

	int AddTexture(const Texture& texture);
	void ApplyTextureToMaterial(int materialId, int diffuseMapId);

	bool SendDataToDevice();

	bool IsInvalid() { return m_InvalidMaterials.size() > 0; }

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
	DeviceVector<BVH8, D_BVH8> m_DeviceBvhs;
	DeviceInstance<D_BVH8*> m_DeviceBvhsAddress;
};
