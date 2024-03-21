#include "Scene.h"
#include "Renderer/renderer.cuh"
#include "cuda/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height))
{
	{
		Material material = { make_float3(0.0f, 1.0f, 1.0f) };
		Sphere sphere = {
			0.5f,
			make_float3(-0.8f, 0.0f, 0.0f),
			material
		};
		m_Spheres.push_back(sphere);
	}
	{
		Material material = { make_float3(0.35f, 0.35f, 0.35f) };
		Sphere sphere = {
			99.3f,
			make_float3(0.0f, -100.0f, 0.0f),
			material
		};
		m_Spheres.push_back(sphere);
	}
	{
		Material material = { make_float3(0.0f, 1.0f, 0.0f) };
		Sphere sphere = {
			0.5f,
			make_float3(0.8f, 0.0f, 0.0f),
			material
		};
		m_Spheres.push_back(sphere);
	}
}

void Scene::SendDataToDevice()
{
	m_Invalid = false;
	SendSceneDataToDevice(this);
}
