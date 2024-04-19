#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	SceneType sceneType = SceneType::NEONS;

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

		assetManager.AddMesh("assets/models/plane.obj");
		assetManager.AddMesh("assets/models/cube.obj");

		MeshInstance& ground = m_Scene.CreateMeshInstance(0);
		ground.SetScale(4.0f);
		ground.AssignMaterial(0);
		MeshInstance& backWall = m_Scene.CreateMeshInstance(0);
		backWall.AssignMaterial(0);
		backWall.SetScale(4.0f);
		backWall.SetRotationX(90.0f);
		backWall.SetPosition(make_float3(0.0f, 4.0f, -4.0f));
		MeshInstance& roof = m_Scene.CreateMeshInstance(0);
		roof.SetScale(4.0f);
		roof.SetRotationX(180.0f);
		roof.SetPosition(make_float3(0.0f, 8.0f, 0.0f));
		roof.AssignMaterial(0);
		MeshInstance& redWall = m_Scene.CreateMeshInstance(0);
		redWall.SetScale(4.0f);
		redWall.SetRotationZ(-90.0f);
		redWall.SetPosition(make_float3(-4.0f, 4.0f, 0.0f));
		redWall.AssignMaterial(1);
		MeshInstance& greenWall = m_Scene.CreateMeshInstance(0);
		greenWall.SetScale(4.0f);
		greenWall.SetRotationZ(90.0f);
		greenWall.SetPosition(make_float3(4.0f, 4.0f, 0.0f));
		greenWall.AssignMaterial(2);
		MeshInstance& smallCube = m_Scene.CreateMeshInstance(1);
		smallCube.AssignMaterial(3);
		smallCube.SetRotationY(-18.0f);
		smallCube.SetScale(1.2f);
		smallCube.SetPosition(make_float3(1.4f, 1.2f, 1.2f));
		MeshInstance& bigCube = m_Scene.CreateMeshInstance(1);
		bigCube.AssignMaterial(4);
		bigCube.SetRotationY(18.0f);
		bigCube.SetScale(make_float3(1.2f, 2.4f, 1.2f));
		bigCube.SetPosition(make_float3(-1.3f, 2.4f, -1.3f));
		MeshInstance& light = m_Scene.CreateMeshInstance(0);
		light.SetRotationX(180.0f);
		light.SetPosition(make_float3(0.0f, 7.99f, 0.0f));
		light.AssignMaterial(5);
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

		assetManager.AddMesh("assets/models/plane.obj");
		assetManager.AddMesh("assets/models/sphere.obj");

		MeshInstance& ground = m_Scene.CreateMeshInstance(0);
		ground.SetScale(4.0f);
		ground.AssignMaterial(0);
		MeshInstance& backWall = m_Scene.CreateMeshInstance(0);
		backWall.AssignMaterial(0);
		backWall.SetScale(4.0f);
		backWall.SetRotationX(90.0f);
		backWall.SetPosition(make_float3(0.0f, 4.0f, -4.0f));
		MeshInstance& roof = m_Scene.CreateMeshInstance(0);
		roof.SetScale(4.0f);
		roof.SetRotationX(180.0f);
		roof.SetPosition(make_float3(0.0f, 8.0f, 0.0f));
		roof.AssignMaterial(0);
		MeshInstance& redWall = m_Scene.CreateMeshInstance(0);
		redWall.SetScale(4.0f);
		redWall.SetRotationZ(-90.0f);
		redWall.SetPosition(make_float3(-4.0f, 4.0f, 0.0f));
		redWall.AssignMaterial(1);
		MeshInstance& greenWall = m_Scene.CreateMeshInstance(0);
		greenWall.SetScale(4.0f);
		greenWall.SetRotationZ(90.0f);
		greenWall.SetPosition(make_float3(4.0f, 4.0f, 0.0f));
		greenWall.AssignMaterial(2);
		MeshInstance& sphere1 = m_Scene.CreateMeshInstance(1);
		sphere1.AssignMaterial(3);
		sphere1.SetScale(1.3f);
		sphere1.SetPosition(make_float3(1.8f, 1.3f, 1.3f));
		MeshInstance& sphere2 = m_Scene.CreateMeshInstance(1);
		sphere2.AssignMaterial(4);
		sphere2.SetScale(1.3f);
		sphere2.SetPosition(make_float3(-1.8f, 1.3f, -0.9f));
		MeshInstance& light = m_Scene.CreateMeshInstance(0);
		light.SetRotationX(180.0f);
		light.SetPosition(make_float3(0.0f, 7.99f, 0.0f));
		light.AssignMaterial(5);
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

		assetManager.AddMesh("assets/models/plane.obj");
		assetManager.AddMesh("assets/models/dragon.obj");
		assetManager.AddMesh("assets/models/cube.obj");
		assetManager.AddMesh("assets/models/light2.obj");

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
		material.diffuse.albedo = make_float3(0.3f, 0.0f, 0.0f);
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
		material.light.emission = make_float3(15.0f, 15.0f, 15.0f);
		assetManager.AddMaterial(material);

		assetManager.AddMesh("assets/models/plane.obj");
		assetManager.AddMesh("assets/models/sphere.obj");

		MeshInstance& floor = m_Scene.CreateMeshInstance(0);
		floor.AssignMaterial(0);
		floor.SetScale(100.0f);
		MeshInstance& sphere1 = m_Scene.CreateMeshInstance(1);
		sphere1.AssignMaterial(3);
		sphere1.SetScale(1.3f);
		sphere1.SetPosition(make_float3(0.0f, 1.3f, 0.0f));
		MeshInstance& light1 = m_Scene.CreateMeshInstance(0);
		light1.SetScale(make_float3(0.5f, 1.0f, 5.0f));
		light1.SetRotationX(180.0f);
		light1.SetPosition(make_float3(-8.0f, 8.0f, 0.0f));
		light1.AssignMaterial(5);
		MeshInstance& light2 = m_Scene.CreateMeshInstance(0);
		light2.SetScale(make_float3(0.5f, 1.0f, 5.0f));
		light2.SetRotationX(180.0f);
		light2.SetPosition(make_float3(-4.0f, 8.0f, 0.0f));
		light2.AssignMaterial(5);
		MeshInstance& light3 = m_Scene.CreateMeshInstance(0);
		light3.SetScale(make_float3(0.5f, 1.0f, 5.0f));
		light3.SetRotationX(180.0f);
		light3.SetPosition(make_float3(0.0f, 8.0f, 0.0f));
		light3.AssignMaterial(5);
		MeshInstance& light4 = m_Scene.CreateMeshInstance(0);
		light4.SetScale(make_float3(0.5f, 1.0f, 5.0f));
		light4.SetRotationX(180.0f);
		light4.SetPosition(make_float3(4.0f, 8.0f, 0.0f));
		light4.AssignMaterial(5);
		MeshInstance& light5 = m_Scene.CreateMeshInstance(0);
		light5.SetScale(make_float3(0.5f, 1.0f, 5.0f));
		light5.SetRotationX(180.0f);
		light5.SetPosition(make_float3(8.0f, 8.0f, 0.0f));
		light5.AssignMaterial(5);
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

