#include "Scene.h"
#include "Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 4.0f, 14.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height, 5.0f, 0.0f))
{
}

void Scene::AddMaterial(Material& material)
{
	m_AssetManager.AddMaterial(material);
}

void Scene::SendDataToDevice()
{
	m_Invalid = false;
	SendSceneDataToDevice(this);
}
