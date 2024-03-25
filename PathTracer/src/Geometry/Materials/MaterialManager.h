#pragma once

#include <vector>
#include <map>
#include "Material.h"

class MaterialManager
{
public:
	MaterialManager() = default;
	~MaterialManager();

	std::vector<Material*>& GetMaterials() { return m_Materials; };
	Material* GetDevicePtr(uint32_t id) { return m_DevicePtr[id]; };
	Material& GetMaterialForPtr(Material* material) { return *m_Materials[m_IdForDevicePtr[material]]; };

	void AddMaterial(Material* material);
	void Invalidate(uint32_t id);
	bool SendDataToDevice();

private:
	std::vector<Material*> m_Materials;
	std::vector<Material*> m_DevicePtr;
	std::map<Material*, uint32_t> m_IdForDevicePtr;
	std::vector<uint32_t> m_InvalidMaterials;
};
