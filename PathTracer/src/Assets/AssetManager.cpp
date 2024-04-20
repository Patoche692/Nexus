#include "AssetManager.h"
#include "OBJLoader.h"
#include "IMGLoader.h"

AssetManager::AssetManager()
{
	m_MaterialSymbolAddress = getMaterialSymbolAddress();
	m_MeshSymbolAddress = getMeshSymbolAddress();
}

AssetManager::~AssetManager()
{
	freeDeviceMaterials();
	freeDeviceTLAS();
	for (BVH* bvh : m_Bvh)
		delete bvh;
}

void AssetManager::AddMesh(const std::string& filename)
{
	std::vector<Mesh> meshes = OBJLoader::LoadOBJ(filename, this);
	for (Mesh& mesh : meshes)
		m_Meshes.push_back(mesh);
}

void AssetManager::AddMaterial()
{
	Material material;
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.2f, 0.2f, 0.2f);
	AddMaterial(material);
}

int AssetManager::AddMaterial(const Material& material)
{
	m_Materials.push_back(material);
	Material& m = m_Materials[m_Materials.size() - 1];
	newDeviceMaterial(m, m_Materials.size());
	return m_Materials.size() - 1;
}

void AssetManager::InvalidateMaterial(uint32_t index)
{
	m_InvalidMaterials.insert(index);
}

int AssetManager::AddTexture(const std::string& filename)
{
	Texture newTexture = IMGLoader::LoadIMG(filename);
	m_Textures.push_back(newTexture);
	Texture& m = m_Textures[m_Textures.size() - 1];
	newDeviceTexture(m, m_Textures.size());
	return m_Textures.size() - 1;
}

void AssetManager::ApplyTextureToMaterial(int materialId, int textureId)
{
	m_Materials[materialId].textureId = textureId;
	InvalidateMaterial(materialId);
}

bool AssetManager::SendDataToDevice()
{
	bool invalid = false;
	for (uint32_t id : m_InvalidMaterials)
	{
		invalid = true;
		cpyMaterialToDevice(m_Materials[id], id);
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
	materialTypes.append("Light");
	materialTypes.push_back('\0');
	materialTypes.append("Diffuse");
	materialTypes.push_back('\0');
	materialTypes.append("Plastic");
	materialTypes.push_back('\0');
	materialTypes.append("Dielectric");
	materialTypes.push_back('\0');
	materialTypes.append("Conductor");
	materialTypes.push_back('\0');
	return materialTypes;
}
