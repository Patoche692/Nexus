#include "RayTracerApplication.h"

RayTracerApplication::RayTracerApplication(int width, int height, GLFWwindow *window)
	:m_Renderer(width, height, window), m_Scene(width, height)
{
	//SceneType sceneType = SceneType::DRAGONS;
	SceneType sceneType = SceneType::CORNELL_BOX;				// change scene

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
		assetManager.AddMesh("assets/models/lego.obj");
		assetManager.AddMesh("assets/models/light.obj");

		//assetManager.AddTexture("assets/textures/brickwall.jpg");
		//assetManager.ApplyTextureToMaterial(0, 0);


		MeshInstance& greyFaces = m_Scene.CreateMeshInstance(0);
		greyFaces.AssignMaterial(0);
		MeshInstance& redFace = m_Scene.CreateMeshInstance(1);
		redFace.AssignMaterial(1);
		MeshInstance& greenFace = m_Scene.CreateMeshInstance(2);
		greenFace.AssignMaterial(2);
		m_Scene.CreateMeshInstance(3);
		m_Scene.CreateMeshInstance(4);
		m_Scene.CreateMeshInstance(5);
		m_Scene.CreateMeshInstance(6);
		m_Scene.CreateMeshInstance(7);
		m_Scene.CreateMeshInstance(8);
		m_Scene.CreateMeshInstance(9);
		m_Scene.CreateMeshInstance(10);
		m_Scene.CreateMeshInstance(11);
		MeshInstance& light = m_Scene.CreateMeshInstance(12);
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

		assetManager.AddMesh("assets/models/box_grey_faces.obj");
		assetManager.AddMesh("assets/models/box_red_face.obj");
		assetManager.AddMesh("assets/models/box_green_face.obj");
		assetManager.AddMesh("assets/models/sphere.obj");
		assetManager.AddMesh("assets/models/light.obj");

		MeshInstance& greyFaces = m_Scene.CreateMeshInstance(0);
		greyFaces.AssignMaterial(0);
		MeshInstance& redFace = m_Scene.CreateMeshInstance(1);
		redFace.AssignMaterial(1);
		MeshInstance& greenFace = m_Scene.CreateMeshInstance(2);
		greenFace.AssignMaterial(2);
		MeshInstance& sphere1 = m_Scene.CreateMeshInstance(3);
		sphere1.AssignMaterial(3);
		sphere1.SetScale(1.3f);
		sphere1.SetPosition(make_float3(1.8f, 1.3f, 1.3f));
		MeshInstance& sphere2 = m_Scene.CreateMeshInstance(3);
		sphere2.AssignMaterial(4);
		sphere2.SetScale(1.3f);
		sphere2.SetPosition(make_float3(-1.8f, 1.3f, -0.9f));
		MeshInstance& light = m_Scene.CreateMeshInstance(4);
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

