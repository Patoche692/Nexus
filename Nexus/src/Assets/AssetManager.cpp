#include "AssetManager.h"
#include "OBJLoader.h"
#include "IMGLoader.h"

void AssetManager::Reset()
{
	m_Materials.clear();
	m_DeviceMaterials.clear();
	m_InvalidMaterials.clear();
	m_DiffuseMaps.clear();
	m_DeviceDiffuseMaps.clear();
	m_EmissiveMaps.clear();
	m_DeviceEmissiveMaps.clear();
	m_Meshes.clear();
}

void AssetManager::AddMesh(const std::string& path, const std::string filename)
{
	std::vector<Mesh> meshes = OBJLoader::LoadOBJ(path, filename, this);
	for (Mesh& mesh : meshes)
		m_Meshes.push_back(mesh);
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
	m_DeviceMaterials.push_back(material.ToDevice());
	Material& m = m_Materials[m_Materials.size() - 1];
	return m_Materials.size() - 1;
}

void AssetManager::InvalidateMaterial(uint32_t index)
{
	m_InvalidMaterials.insert(index);
}

int AssetManager::AddTexture(Texture& texture)
{
	if (texture.pixels == NULL)
	{
		return -1;
	}

	if (texture.type == Texture::Type::DIFFUSE)
	{
		m_DiffuseMaps.push_back(texture);
		m_DeviceDiffuseMaps.push_back(texture.ToDevice());
		Texture& m = m_DiffuseMaps[m_DiffuseMaps.size() - 1];
		return m_DiffuseMaps.size() - 1;
	}
	else if (texture.type == Texture::Type::EMISSIVE)
	{
		m_EmissiveMaps.push_back(texture);
		m_DeviceEmissiveMaps.push_back(texture.ToDevice());
		Texture& m = m_EmissiveMaps[m_EmissiveMaps.size() - 1];
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
		m_DeviceMaterials[id] = m_Materials[id].ToDevice();
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
