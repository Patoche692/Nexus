#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	SceneType scene = SceneType::CORNELL_BOX_SPHERES;

	AssetManager& assetManager = m_Scene.GetAssetManager();

	if (scene == SceneType::CORNELL_BOX)
	{
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
		material.diffuse.albedo = make_float3(0.93f, 0.93f, 0.93f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIFFUSE;
		material.diffuse.albedo = make_float3(0.93f, 0.93f, 0.93f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::LIGHT;
		material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/box_grey_faces.obj", 0);
		assetManager.AddMesh("assets/models/box_red_face.obj", 1);
		assetManager.AddMesh("assets/models/box_green_face.obj", 2);
		assetManager.AddMesh("assets/models/cube.obj", 3);
		assetManager.AddMesh("assets/models/light.obj", 5);

		Mat4 transform = Mat4::Identity();
		assetManager.CreateInstance(0, transform);
		assetManager.CreateInstance(1, transform);
		assetManager.CreateInstance(2, transform);
		transform = Mat4::Scale(1.2f) * Mat4::Translate(make_float3(1.1f, 1.0f, 1.1f)) * Mat4::RotateY(-0.3f);
		assetManager.CreateInstance(3, transform);
		transform =  Mat4::Scale(make_float3(1.2f, 2.4f, 1.2f)) * Mat4::Translate(make_float3(-1.1f, 1.0f, -0.8f)) * Mat4::RotateY(0.3f);
		assetManager.CreateInstance(3, transform);
		transform = Mat4::Identity();
		assetManager.CreateInstance(4, transform);

		assetManager.BuildTLAS();
	}
	else if (scene == SceneType::CORNELL_BOX_SPHERES)
	{
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
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(1.0f);
		material.plastic.roughness = 0.0f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::LIGHT;
		material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/box_grey_faces.obj", 0);
		assetManager.AddMesh("assets/models/box_red_face.obj", 1);
		assetManager.AddMesh("assets/models/box_green_face.obj", 2);
		assetManager.AddMesh("assets/models/sphere.obj", 3);
		assetManager.AddMesh("assets/models/light.obj", 4);

		Mat4 transform = Mat4::Identity();
		assetManager.CreateInstance(0, transform);
		assetManager.CreateInstance(1, transform);
		assetManager.CreateInstance(2, transform);
		transform = Mat4::Scale(1.3f) * Mat4::Translate(make_float3(1.4f, 1.0f, 0.9f));
		assetManager.CreateInstance(3, transform);
		transform = Mat4::Scale(1.3f) * Mat4::Translate(make_float3(-1.4f, 1.0f, -0.6f));
		assetManager.CreateInstance(3, transform);
		transform = Mat4::Identity();
		assetManager.CreateInstance(4, transform);

		assetManager.BuildTLAS();

	}
	else if (scene == SceneType::DRAGON)
	{
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
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(1.0f, 0.5f, 0.0f);
		material.plastic.roughness = 0.15f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::LIGHT;
		material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/box_grey_faces.obj", 0);
		assetManager.AddMesh("assets/models/box_red_face.obj", 1);
		assetManager.AddMesh("assets/models/box_green_face.obj", 2);
		assetManager.AddMesh("assets/models/dragon.obj", 3);
		assetManager.AddMesh("assets/models/light.obj", 4);

		Mat4 transform = Mat4::Identity();
		assetManager.CreateInstance(0, transform);
		assetManager.CreateInstance(1, transform);
		assetManager.CreateInstance(2, transform);
		transform = Mat4::Translate(make_float3(1.5f, 1.2f, 0.8f)) * Mat4::RotateY(1.8f) * Mat4::Scale(4.0f);
		assetManager.CreateInstance(3, transform);
		transform = Mat4::Translate(make_float3(-1.5f, 1.0f, -0.8f)) * Mat4::RotateY(-1.8) * Mat4::Scale(3.0f);
		assetManager.CreateInstance(3, transform);
		transform = Mat4::Identity();
		assetManager.CreateInstance(4, transform);

		assetManager.BuildTLAS();

	}

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

