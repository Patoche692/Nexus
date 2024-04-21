#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	//SceneType sceneType = SceneType::DRAGONS;
	SceneType sceneType = SceneType::LIVING_ROOM;				// change scene

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

