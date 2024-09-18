#include "AssetManager.h"
#include "OBJLoader.h"
#include "IMGLoader.h"
#include "Cuda/PathTracer/PathTracer.cuh"

AssetManager::AssetManager()
	: m_DeviceBvhsAddress(GetDeviceBVHAddress())
{
}

void AssetManager::Reset()
{
	m_Materials.clear();
	m_InvalidMaterials.clear();
	m_DiffuseMaps.clear();
	m_EmissiveMaps.clear();
	m_DeviceDiffuseMaps.Clear();
	m_DeviceEmissiveMaps.Clear();
	m_DeviceMaterials.Clear();
	m_Meshes.clear();
}

int32_t AssetManager::CreateBVH(const std::vector<Triangle>& triangles)
{
	BVH8 bvh8;
	std::cout << "Triangle count: " << triangles.size() << std::endl;

	BVH8Builder builder(triangles);
	builder.Init();
	bvh8 = builder.Build();
	bvh8.InitDeviceData();

	std::cout << "BVH8 successfully built" << std::endl << std::endl;

	m_Bvhs.push_back(std::move(bvh8));
	return m_Bvhs.size() - 1;
}

int32_t  AssetManager::AddMesh(Mesh&& mesh)
{
	m_Meshes.push_back(std::move(mesh));
	return m_Meshes.size() - 1;
}

void AssetManager::InitDeviceData()
{
	m_DeviceBvhs = m_Bvhs;
	m_DeviceBvhsAddress = m_DeviceBvhs.Data();
}

void AssetManager::AddMaterial()
{
	Material material;
	material.diffuse.albedo = make_float3(0.2f, 0.2f, 0.2f);
	AddMaterial(material);
}

int AssetManager::AddMaterial(const Material& material)
{
	m_Materials.push_back(material);
	m_DeviceMaterials.PushBack(material);
	Material& m = m_Materials[m_Materials.size() - 1];
	return m_Materials.size() - 1;
}

void AssetManager::InvalidateMaterial(uint32_t index)
{
	m_InvalidMaterials.insert(index);
}

int AssetManager::AddTexture(const Texture& texture)
{
	if (texture.pixels == NULL)
	{
		return -1;
	}

	if (texture.type == Texture::Type::DIFFUSE)
	{
		m_DiffuseMaps.push_back(texture);
		m_DeviceDiffuseMaps.PushBack(texture);
		return m_DiffuseMaps.size() - 1;
	}
	else if (texture.type == Texture::Type::EMISSIVE)
	{
		m_EmissiveMaps.push_back(texture);
		m_DeviceEmissiveMaps.PushBack(texture);
		return m_EmissiveMaps.size() - 1;
	}
}

void AssetManager::ApplyTextureToMaterial(int materialId, int diffuseMapId)
{
	m_Materials[materialId].diffuseMapId = diffuseMapId;
	InvalidateMaterial(materialId);
}

bool AssetManager::SendDataToDevice()
{
	bool invalid = false;
	for (uint32_t id : m_InvalidMaterials)
	{
		invalid = true;
		m_DeviceMaterials[id] = m_Materials[id];
	}
	m_InvalidMaterials.clear();
	return invalid;
}

std::string AssetManager::GetMaterialsString()
{
	std::string materialsString;
	for (int i = 0; i < m_Materials.size(); i++)
	{
		materialsString.append("Material ");
		materialsString.append(std::to_string(i));
		materialsString.push_back('\0');
	}
	return materialsString;
}

std::string AssetManager::GetMaterialTypesString()
{
	std::string materialTypes;
	materialTypes.append("Diffuse");
	materialTypes.push_back('\0');
	materialTypes.append("Dielectric");
	materialTypes.push_back('\0');
	materialTypes.append("Conductor");
	materialTypes.push_back('\0');
	return materialTypes;
}
