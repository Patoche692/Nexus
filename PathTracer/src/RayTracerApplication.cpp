#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	SceneType sceneType = SceneType::CORNELL_BOX_SPHERES;

	AssetManager& assetManager = m_Scene.GetAssetManager();

	if (sceneType == SceneType::CORNELL_BOX)
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

		assetManager.AddMesh("assets/models/box_grey_faces.obj");
		assetManager.AddMesh("assets/models/box_red_face.obj");
		assetManager.AddMesh("assets/models/box_green_face.obj");
		assetManager.AddMesh("assets/models/cube.obj");
		assetManager.AddMesh("assets/models/light.obj");

		m_Scene.CreateMeshInstance(0);
		m_Scene.CreateMeshInstance(1);
		m_Scene.CreateMeshInstance(2);

		MeshInstance& instance = m_Scene.CreateMeshInstance(3);
		instance.Scale(1.2f);
		instance.Translate(make_float3(1.1f, 1.0f, 1.1f));
		instance.RotateY(-18.0f);

		instance = m_Scene.CreateMeshInstance(3);
		instance.Scale(make_float3(1.2f, 2.4f, 1.2f));
		instance.Translate(make_float3(-1.1f, 1.0f, -0.8f));
		instance.RotateY(18.0f);

		m_Scene.CreateMeshInstance(4);

		//transform = Mat4::Scale(1.2f) * Mat4::Translate(make_float3(1.1f, 1.0f, 1.1f)) * Mat4::RotateY(-0.3f);
		//assetManager.CreateMeshInstance(3, transform);
		//transform =  Mat4::Scale(make_float3(1.2f, 2.4f, 1.2f)) * Mat4::Translate(make_float3(-1.1f, 1.0f, -0.8f)) * Mat4::RotateY(0.3f);
		//assetManager.CreateMeshInstance(3, transform);
		//transform = Mat4::Identity();
		//assetManager.CreateMeshInstance(4, transform);
	}
	else if (sceneType == SceneType::CORNELL_BOX_SPHERES)
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

		assetManager.AddMesh("assets/models/box_grey_faces.obj");
		assetManager.AddMesh("assets/models/box_red_face.obj");
		assetManager.AddMesh("assets/models/box_green_face.obj");
		assetManager.AddMesh("assets/models/sphere.obj");
		assetManager.AddMesh("assets/models/light.obj");

		m_Scene.CreateMeshInstance(0);
		m_Scene.CreateMeshInstance(1);
		m_Scene.CreateMeshInstance(2);

		MeshInstance& instance = m_Scene.CreateMeshInstance(3);
		instance.Scale(1.3f);
		instance.Translate(make_float3(1.4f, 1.0f, 0.9f));

		instance = m_Scene.CreateMeshInstance(3);
		instance.Scale(1.3f);
		instance.Translate(make_float3(-1.4f, 1.0f, -0.6f));

		m_Scene.CreateMeshInstance(4);
	}
	else if (sceneType == SceneType::DRAGON)
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

		assetManager.AddMesh("assets/models/box_grey_faces.obj");
		assetManager.AddMesh("assets/models/box_red_face.obj");
		assetManager.AddMesh("assets/models/box_green_face.obj");
		assetManager.AddMesh("assets/models/dragon.obj");
		assetManager.AddMesh("assets/models/light.obj");

		m_Scene.CreateMeshInstance(0);
		m_Scene.CreateMeshInstance(1);
		m_Scene.CreateMeshInstance(2);

		MeshInstance& instance = m_Scene.CreateMeshInstance(3);
		instance.Translate(make_float3(1.5f, 1.2f, 0.8f));
		instance.RotateY(100.0f);
		instance.Scale(4.0f);

		instance = m_Scene.CreateMeshInstance(3);
		instance.Translate(make_float3(-1.5f, 1.0f, -0.8f));
		instance.RotateY(-100.0f);
		instance.Scale(3.0f);

		m_Scene.CreateMeshInstance(4);
	}
	m_Scene.BuildTLAS();
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

