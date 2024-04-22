#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	SceneType sceneType = SceneType::ELLIE;

	AssetManager& assetManager = m_Scene.GetAssetManager();

	if (sceneType == SceneType::CORNELL_BOX)
	{
		assetManager.AddMesh("assets/scenes/cornell_box/", "cornell_box.obj");

		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::CORNELL_BOX_SPHERE)
	{
		assetManager.AddMesh("assets/scenes/cornell_box_sphere/", "cornell_box_sphere.obj");
		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::DINING_ROOM)
	{
		assetManager.AddMesh("assets/scenes/dining_room/", "dining_room.obj");
		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::LIVING_ROOM)
	{
		assetManager.AddMesh("assets/scenes/living_room/", "living_room.obj");
		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::BATHROOM)
	{
		assetManager.AddMesh("assets/scenes/bathroom/", "bathroom.obj");
		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::ELLIE)
	{
		assetManager.AddMesh("assets/scenes/ellie/", "ellie.obj");
		for (int i = 0; i < assetManager.GetMeshes().size(); i++)
			m_Scene.CreateMeshInstance(i);
	}
	else if (sceneType == SceneType::DRAGONS)
	{
		Material material;
		material.type = Material::Type::DIFFUSE;
		material.diffuse.albedo = make_float3(1.0f, 1.0f, 1.0f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIFFUSE;
		material.diffuse.albedo = make_float3(0.85f, 0.05f, 0.05f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIELECTRIC;
		material.dielectric.ior = 1.0f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIELECTRIC;
		material.dielectric.ior = 1.5f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(1.0f, 0.5f, 0.0f);
		material.plastic.roughness = 0.15f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::LIGHT;
		material.light.emission = make_float3(35.0f, 35.0f, 35.0f);
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/", "plane.obj");
		assetManager.AddMesh("assets/models/", "dragon.obj");
		assetManager.AddMesh("assets/models/", "cube.obj");
		assetManager.AddMesh("assets/models/", "light2.obj");

		MeshInstance& greyFaces = m_Scene.CreateMeshInstance(0);
		greyFaces.AssignMaterial(0);
		greyFaces.SetScale(100.0f);
		MeshInstance& dragon1 = m_Scene.CreateMeshInstance(1);
		dragon1.AssignMaterial(3);
		dragon1.SetRotationY(100.0f);
		dragon1.SetScale(4.0f);
		dragon1.SetPosition(make_float3(8.0f, 1.2f, 1.2f));
		MeshInstance& dragon2 = m_Scene.CreateMeshInstance(1);
		dragon2.AssignMaterial(4);
		dragon2.SetRotationY(90.0f);
		dragon2.SetScale(3.0f);
		dragon2.SetPosition(make_float3(0.0f, 1.0f, 0.5f));
		MeshInstance& container = m_Scene.CreateMeshInstance(2);
		container.AssignMaterial(2);
		container.SetScale(make_float3(1.6f, 1.2f, 1.2f));
		container.SetPosition(make_float3(0.0f, 1.21f, 0.5f));

		MeshInstance& light = m_Scene.CreateMeshInstance(3);
		light.AssignMaterial(5);
	}
	else if (sceneType == SceneType::NEONS)
	{
		Material material;
		material.type = Material::Type::DIFFUSE;
		material.diffuse.albedo = make_float3(0.7f, 0.0f, 0.0f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIFFUSE;
		material.diffuse.albedo = make_float3(0.85f, 0.05f, 0.05f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(0.8f);
		material.plastic.roughness = 0.0f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::DIELECTRIC;
		material.dielectric.ior = 1.5f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(1.0f, 0.5f, 0.0f);
		material.plastic.roughness = 0.15f;
		assetManager.AddMaterial(material);
		material.type = Material::Type::LIGHT;
		material.light.emission = make_float3(7.0f, 7.0f, 7.0f);
		assetManager.AddMaterial(material);
		material.type = Material::Type::METAL;
		material.plastic.albedo = make_float3(0.8f, 0.2f, 0.2f);
		material.plastic.roughness = 0.05f;
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/", "plane.obj");
		assetManager.AddMesh("assets/models/", "sphere.obj");
		assetManager.AddMesh("assets/models/", "cube.obj");
		assetManager.AddMesh("assets/models/", "ring.obj");

		MeshInstance& floor = m_Scene.CreateMeshInstance(0);
		floor.AssignMaterial(0);
		floor.SetScale(100.0f);
		floor.SetPosition(make_float3(0.0f, -0.5f, 0.0f));
		MeshInstance& stair = m_Scene.CreateMeshInstance(2);
		stair.AssignMaterial(0);
		stair.SetScale(make_float3(10.0f, 0.15f, 6.0f));
		stair.SetPosition(make_float3(0.0f, -0.15f, 0.0f));
		MeshInstance& sphere1 = m_Scene.CreateMeshInstance(1);
		sphere1.AssignMaterial(3);
		sphere1.SetScale(1.0f);
		sphere1.SetPosition(make_float3(7.0f, 1.0f, 0.0f));
		MeshInstance& sphere2 = m_Scene.CreateMeshInstance(1);
		sphere2.AssignMaterial(6);
		sphere2.SetScale(0.8f);
		sphere2.SetPosition(make_float3(3.5f, 0.8f, -2.0f));
		MeshInstance& sphere3 = m_Scene.CreateMeshInstance(1);
		sphere3.AssignMaterial(2);
		sphere3.SetScale(1.0f);
		sphere3.SetPosition(make_float3(-7.0f, 1.0f, 0.5f));
		MeshInstance& sphere4 = m_Scene.CreateMeshInstance(1);
		sphere4.AssignMaterial(6);
		sphere4.SetScale(0.7f);
		sphere4.SetPosition(make_float3(-3.5f, 0.7f, 2.0f));
		MeshInstance& sphere5 = m_Scene.CreateMeshInstance(1);
		sphere5.AssignMaterial(6);
		sphere5.SetScale(0.2f);
		sphere5.SetPosition(make_float3(0.2f, 0.2f, 5.0f));
		MeshInstance& ring1 = m_Scene.CreateMeshInstance(3);
		ring1.AssignMaterial(2);
		ring1.SetScale(make_float3(0.5f, 0.7f, 0.5f));
		ring1.SetPosition(make_float3(0.0f, 0.0f, 0.0f));
		MeshInstance& ring2 = m_Scene.CreateMeshInstance(3);
		ring2.AssignMaterial(2);
		ring2.SetScale(make_float3(0.5f, 0.7f, 0.5f));
		ring2.SetPosition(make_float3(1.0f, 0.0f, 1.0f));
		MeshInstance& ring3 = m_Scene.CreateMeshInstance(3);
		ring3.AssignMaterial(2);
		ring3.SetScale(make_float3(0.5f, 0.7f, 0.5f));
		ring3.SetPosition(make_float3(-2.0f, 0.0f, -0.4f));
		for (int i = 0; i < 16; i++)
		{
			MeshInstance& light = m_Scene.CreateMeshInstance(0);
			light.SetScale(make_float3(0.25f, 1.0f, 2.0f));
			light.SetRotationX(180.0f);
			light.SetPosition(make_float3(-8.0f + i, 4.0f, 0.0f));
			light.AssignMaterial(5);
		}
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

