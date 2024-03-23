#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	Material* material = new Material(make_float3(0.0f, 1.0f, 1.0f));
	m_Materials.push_back(material);
	Sphere sphere = {
		0.5f,
		make_float3(-0.8f, 0.0f, 0.0f),
		*material
	};
	m_Scene.AddSphere(sphere);

	material = new Material( make_float3(0.35f, 0.35f, 0.35f));
	m_Materials.push_back(material);
	sphere = {
		9999.3f,
		make_float3(0.0f, -10000.0f, 0.0f),
		*material
	};
	m_Scene.AddSphere(sphere);

	material = new Material(make_float3(0.0f, 1.0f, 0.0f));
	m_Materials.push_back(material);
	sphere = {
		0.5f,
		make_float3(0.8f, 0.0f, 0.0f),
		*material
	};
	m_Scene.AddSphere(sphere);
}

RayTracerApplication::~RayTracerApplication()
{
	for (Material* material : m_Materials)
		delete material;
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

