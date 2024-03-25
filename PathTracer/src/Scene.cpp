#include "Scene.h"
#include "../Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 0.0f, 2.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height))
{
}

void Scene::AddSphere()
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	m_Materials.push_back(Material(make_float3(1.0f)));
	AddSphere(Sphere(0.5, make_float3(0.0f), m_Materials[m_Materials.size() - 1]));
}

void Scene::AddSphere(Sphere sphere)
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	m_Spheres.push_back(sphere);
	m_Invalid = true;
}

void Scene::AddMaterial(Material material)
{
	m_Materials.push_back(material);
	m_Invalid = true;
}


void Scene::SendDataToDevice()
{
	m_Invalid = false;
	SendSceneDataToDevice(this);
}
