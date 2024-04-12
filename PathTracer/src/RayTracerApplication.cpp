#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{

	AssetManager& assetManager = m_Scene.GetAssetManager();
	assetManager.AddMesh("assets/models/box_grey_faces.obj");
	assetManager.AddMesh("assets/models/box_red_face.obj");
	assetManager.AddMesh("assets/models/box_green_face.obj");
	assetManager.AddMesh("assets/models/cube.obj");
	assetManager.AddMesh("assets/models/cube2.obj");
	assetManager.AddMesh("assets/models/light.obj");
	Material material;
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.93f, 0.93f, 0.93f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.85f, 0.05f, 0.05f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.12f, 0.75f, 0.15f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.73f, 0.73f, 0.73f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.73f, 0.73f, 0.73f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::LIGHT;
	material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.plastic.albedo = make_float3(0.5f, 0.0f, 0.5f);
	material.plastic.roughness = 0.5f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::METAL;
	material.plastic.albedo = make_float3(1.0f, 1.0f, 1.0f);
	material.plastic.roughness = 0.2f;
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

