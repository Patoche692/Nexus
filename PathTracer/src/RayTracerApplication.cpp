#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{

	AssetManager& assetManager = m_Scene.GetAssetManager();
	assetManager.AddMesh("assets/models/floor.obj");
	assetManager.AddMesh("assets/models/cube.obj");
	Material material;
	material.type = Material::Type::PLASTIC;
	material.plastic.albedo = make_float3(0.07f, 0.07f, 0.07f);
	material.plastic.roughness = 0.3f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::PLASTIC;
	material.plastic.albedo = make_float3(1.0f, 0.2f, 0.0f);
	material.plastic.roughness = 0.9f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.plastic.albedo = make_float3(0.5f, 0.0f, 0.5f);
	material.plastic.roughness = 0.5f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::PLASTIC;
	material.plastic.albedo = make_float3(1.0f, 1.0f, 1.0f);
	material.plastic.roughness = 0.2f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIELECTRIC;
	material.dielectric.ir = 1.3f;
	material.dielectric.roughness = 0.9f;
	assetManager.AddMaterial(material);

	Sphere sphere = {
		1000.0f,
		make_float3(0.0f, -1000.0f, 0.0f),
		0
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.9f,
		make_float3(0.0f, 0.9f, 0.0f),
		1
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.62f,
		make_float3(1.4f, 0.5f, 0.0f),
		2
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.5f,
		make_float3(-1.4f, 0.5f, 0.0f),
		3
	};
	m_Scene.AddSphere(sphere);

	sphere = {
		0.5f,
		make_float3(0.0f, 0.5f, 1.4f),
		4
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

