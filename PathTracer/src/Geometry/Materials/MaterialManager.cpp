#include <string>
#include <cuda_runtime_api.h>
#include "../../Utils/Utils.h"
#include "MaterialManager.h"
#include "../../Cuda/PathTracer.cuh"


void MaterialManager::AddMaterial()
{
	Material material;
	material.materialType = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.2f, 0.2f, 0.2f);
	AddMaterial(material);
}

void MaterialManager::AddMaterial(Material material)
{
	m_Materials.push_back(material);
	Material& m = m_Materials[m_Materials.size() - 1];
	newDeviceMaterial(m, m_Materials.size());
}

void MaterialManager::Invalidate(uint32_t id)
{
	m_InvalidMaterials.push_back(id);
}

bool MaterialManager::SendDataToDevice()
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

std::string MaterialManager::GetMaterialsString()
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

