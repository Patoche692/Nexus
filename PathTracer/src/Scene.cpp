#include "Scene.h"
#include "../Cuda/Pathtracer.cuh"
#include "Utils/cuda_math.h"

Scene::Scene(uint32_t width, uint32_t height)
	:m_Camera(std::make_shared<Camera>(make_float3(0.0f, 1.0f, 5.0f), make_float3(0.0f, 0.0f, -1.0f), 45.0f, width, height))
{
}

void Scene::AddSphere()
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	AddSphere(Sphere(0.5, make_float3(0.0f), 0));
}

void Scene::AddSphere(Sphere sphere)
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	m_Spheres.push_back(sphere);
	m_Invalid = true;
}

void Scene::AddSphere(int materialId)
{
	if (m_Spheres.size() >= MAX_SPHERES)
		return;

	AddSphere(Sphere(0.5, make_float3(0.0f), 0));
	m_Spheres[m_Spheres.size() - 1].materialId = materialId;
	m_Invalid = true;
}

void Scene::AddMaterial(Material* material)
{
	m_MaterialManager.AddMaterial(*material);
}


void Scene::SendDataToDevice()
{
	m_Invalid = false;
	SendSceneDataToDevice(this);
}
