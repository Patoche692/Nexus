#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{

	AssetManager& assetManager = m_Scene.GetAssetManager();
	assetManager.AddMesh("assets/models/box_grey_faces.obj", 0);
	assetManager.AddMesh("assets/models/box_red_face.obj", 1);
	assetManager.AddMesh("assets/models/box_green_face.obj", 2);
	assetManager.AddMesh("assets/models/sphere.obj", 3);
	assetManager.AddMesh("assets/models/cube2.obj", 4);
	assetManager.AddMesh("assets/models/light.obj", 5);

	Mat4 transform = Mat4::Identity();
	assetManager.CreateInstance(0, transform);
	assetManager.CreateInstance(1, transform);
	assetManager.CreateInstance(2, transform);
	transform = transform.Translate(make_float3(1.0f));
	transform = transform.Scale(make_float3(2.0f));
	assetManager.CreateInstance(3, transform);
	transform = Mat4::Identity();
	assetManager.CreateInstance(4, transform);
	assetManager.CreateInstance(5, transform);

	assetManager.BuildTLAS();

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
	material.type = Material::Type::DIELECTRIC;
	material.dielectric.ior = 1.5f;
	assetManager.AddMaterial(material);
	material.type = Material::Type::DIFFUSE;
	material.diffuse.albedo = make_float3(0.73f, 0.73f, 0.73f);
	assetManager.AddMaterial(material);
	material.type = Material::Type::LIGHT;
	material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
	assetManager.AddMaterial(material);
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

