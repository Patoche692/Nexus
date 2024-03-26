#include <cuda_runtime_api.h>
#include "../../Utils/Utils.h"
#include "MaterialManager.h"
#include "../../Cuda/PathTracer.cuh"


MaterialManager::~MaterialManager()
{
	for (Material* material : m_Materials)
	{
		delete material;
	}
}

void MaterialManager::AddMaterial(Material material)
{
	m_Materials.push_back(new Material(material));
	Material& m = *m_Materials[m_Materials.size() - 1];
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
		//instanciateMaterial(m_DevicePtr[id], *m_Materials[id]);
		changeDeviceMaterial(*m_Materials[id], id);
	}
	m_InvalidMaterials.clear();
	return invalid;
}

