#include "AssetManager.h"
#include "OBJLoader.h"

AssetManager::AssetManager()
{
	m_MaterialSymbolAddress = getMaterialSymbolAddress();
	m_MeshSymbolAddress = getMeshSymbolAddress();
}

AssetManager::~AssetManager()
{
	freeDeviceMaterials();
	freeDeviceMeshes(m_Meshes.size());
	for (Mesh& mesh : m_Meshes)
	{
		delete[] mesh.triangles;
	}
}

void AssetManager::AddMesh(const std::string& filename, int materialId)
{
	Mesh mesh;
	std::vector<Triangle> triangles = OBJLoader::LoadOBJ(filename);

	Triangle* ptr = new Triangle[triangles.size()];
	memcpy(ptr, triangles.data(), triangles.size() * sizeof(Triangle));

	mesh.triangles = ptr;
	mesh.nTriangles = triangles.size();

	if (materialId != -1)
		mesh.materialId = materialId;

	m_Meshes.push_back(mesh);
	newDeviceMesh(mesh, m_Meshes.size());
}

void AssetManager::InvalidateMesh(uint32_t index)
{
	m_InvalidMeshes.insert(index);
}

void AssetManager::InvalidateMeshes()
{
	for (int i = 0; i < m_Meshes.size(); i++)
	{
		m_InvalidMeshes.insert(i);
	}
}

void AssetManager::AddMaterial()
{
	Material material;
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.2f, 0.2f, 0.2f);
	AddMaterial(material);
}

void AssetManager::AddMaterial(const Material& material)
{
	m_Materials.push_back(material);
	Material& m = m_Materials[m_Materials.size() - 1];
	newDeviceMaterial(m, m_Materials.size());
}

void AssetManager::InvalidateMaterial(uint32_t index)
{
	m_InvalidMaterials.insert(index);
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


bool AssetManager::SendDataToDevice()
{
	bool invalid = false;
	for (uint32_t id : m_InvalidMaterials)
	{
		invalid = true;
		changeDeviceMaterial(m_Materials[id], id);
	}
	m_InvalidMaterials.clear();
	return invalid;
}
