#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	Material material(make_float3(0.0f, 1.0f, 1.0f));
	m_Scene.AddMaterial(material);
	material = Material( make_float3(0.35f, 0.35f, 0.35f));
	m_Scene.AddMaterial(material);
	material = Material(make_float3(0.0f, 1.0f, 0.0f));
	m_Scene.AddMaterial(material);

	std::vector<Material>& materials = m_Scene.GetMaterials();

	Sphere sphere = {
		0.5f,
		make_float3(-0.8f, 0.0f, 0.0f),
		materials[0]
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		999.3f,
		make_float3(0.0f, -1000.0f, 0.0f),
		materials[1]
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.5f,
		make_float3(0.8f, 0.0f, 0.0f),
		materials[2]
	};
	m_Scene.AddSphere(sphere);
}

void RayTracerApplication::Update(float deltaTime)
{
	m_Scene.GetCamera()->OnUpdate(deltaTime);
	Display(deltaTime);
}

void RayTracerApplication::Display(float deltaTime)
{
	m_Renderer.Render(m_Scene, deltaTime);
}

void RayTracerApplication::OnResize(int width, int height)
{
}

