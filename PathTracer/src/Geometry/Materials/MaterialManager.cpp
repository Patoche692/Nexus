#include <cuda_runtime_api.h>
#include "../../Utils/Utils.h"
#include "MaterialManager.h"


MaterialManager::~MaterialManager()
{
	for (Material *material : m_DevicePtr)
	{
		checkCudaErrors(cudaFree(material));
	}

	for (Material* material : m_Materials)
	{
		delete material;
	}
}

void MaterialManager::AddMaterial(Material* material)
{
	m_Materials.push_back(material);
	Material& m = *m_Materials[m_Materials.size() - 1];
	m.id = m_Materials.size() - 1;
	m_InvalidMaterials.push_back(m.id);
	m_DevicePtr.push_back(nullptr);
	checkCudaErrors(cudaMalloc((void**)&m_DevicePtr[m.id], m.GetSize()));
	m_IdForDevicePtr[m_DevicePtr[m.id]] = m.id;
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
		checkCudaErrors(cudaMemcpy((void*)m_DevicePtr[id], (void*)m_Materials[id], m_Materials[id]->GetSize(), cudaMemcpyHostToDevice));
	}
	m_InvalidMaterials.clear();
	return invalid;
}

