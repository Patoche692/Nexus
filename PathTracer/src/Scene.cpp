#include "Scene.h"
#include "Renderer/renderer.cuh"
#include "cuda/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height))
{
}

void Scene::AddSphere(Sphere sphere)
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	m_Spheres.push_back(sphere);
	m_Invalid = true;
}

void Scene::SendDataToDevice()
{
	m_Invalid = false;
	SendSceneDataToDevice(this);
}
