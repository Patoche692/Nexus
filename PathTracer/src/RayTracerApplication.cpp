#include "RayTracerApplication.h"
#include "Geometry/Materials/Lambertian.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	MaterialManager& materialManager = m_Scene.GetMaterialManager();
	Material material;
	material.materialType = MaterialType::DIFFUSE;
	material.diffuse.albedo = make_float3(0.0f, 1.0f, 1.0f);
	materialManager.AddMaterial(material);
	material.diffuse.albedo = make_float3(0.35f, 0.35f, 0.35f);
	materialManager.AddMaterial(material);
	material.diffuse.albedo = make_float3(0.0f, 1.0f, 0.0f);
	materialManager.AddMaterial(material);

	std::vector<Material*>& materials = materialManager.GetMaterials();

	Sphere sphere = {
		0.5f,
		make_float3(-0.8f, 0.0f, 0.0f),
		0
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		999.3f,
		make_float3(0.0f, -1000.0f, 0.0f),
		1
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.5f,
		make_float3(0.8f, 0.0f, 0.0f),
		2
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

