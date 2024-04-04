#pragma once

#include <vector>
#include <map>
#include "Material.h"

class MaterialManager
{
public:
	MaterialManager() = default;

	std::vector<Material>& GetMaterials() { return m_Materials; };

	void AddMaterial();
	void AddMaterial(Material material);
	void Invalidate(uint32_t id);
	bool SendDataToDevice();
	std::string GetMaterialsString();

private:
	std::vector<Material> m_Materials;
	std::vector<uint32_t> m_InvalidMaterials;
};
