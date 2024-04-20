#include <gtc/type_ptr.hpp>
#include "Renderer.h"
#include "Cuda/PathTracer.cuh"
#include "Utils/Utils.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


Renderer::Renderer(uint32_t width, uint32_t height, GLFWwindow* window)
	:m_ViewportWidth(width), m_ViewportHeight(height)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsCustomDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");

	m_PixelBuffer = std::make_shared<PixelBuffer>(width, height);
	m_Texture = std::make_shared<OGLTexture>(width, height);

	checkCudaErrors(cudaMalloc((void**)&m_AccumulationBuffer, width * height * sizeof(float3)));

	m_DisplayFPSTimer = glfwGetTime();
}

Renderer::~Renderer()
{

	checkCudaErrors(cudaFree((void*)m_AccumulationBuffer));
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


void Renderer::Render(Scene& scene, float deltaTime)
{ 
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	UpdateTimer(deltaTime);

	// Position UI and resize the texture and pixel buffer depending on the viewport size
	RenderUI(scene);

	if (scene.SendDataToDevice())
		m_FrameNumber = 0;

	m_FrameNumber++;
	// Launch cuda path tracing kernel, writes the viewport into the pixelbuffer
	RenderViewport(m_PixelBuffer, m_FrameNumber, m_AccumulationBuffer);

	// Unpack the pixel buffer written by cuda to the renderer texture
	UnpackToTexture();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::RenderUI(Scene& scene)
{
	ImGui::DockSpaceOverViewport();

	ImGui::Begin("Settings");

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Time info");
	ImGui::Text("Render time millisec: %.3f", m_DeltaTime);
	ImGui::Text("FPS: %d", (int)(1000.0f / m_DeltaTime));
	ImGui::Text("Frame: %d", m_FrameNumber);

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Camera");
	if (ImGui::SliderFloat("Field of view", &scene.GetCamera()->GetVerticalFOV(), 1.0f, 180.0f))
		scene.GetCamera()->Invalidate();
	if (ImGui::DragFloat("Focus distance", &scene.GetCamera()->GetFocusDist(), 0.02f, 0.01f, 1000.0f))
		scene.GetCamera()->Invalidate();
	if (ImGui::DragFloat("Defocus angle", &scene.GetCamera()->GetDefocusAngle(), 0.2f, 0.0f, 180.0f))
		scene.GetCamera()->Invalidate();

	ImGui::End();

	ImGui::Begin("Scene");

	AssetManager& assetManager = scene.GetAssetManager();
	std::vector<Material>& materials = assetManager.GetMaterials();
	std::string materialsString = assetManager.GetMaterialsString();
	std::string materialTypes = assetManager.GetMaterialTypesString();
	std::vector<MeshInstance>& meshInstances = scene.GetMeshInstances();

	for (int i = 0; i < meshInstances.size(); i++)
	{
		MeshInstance& meshInstance = meshInstances[i];
		ImGui::PushID(i);

		//ImGui::ShowDemoWindow();
		if (ImGui::CollapsingHeader("Mesh"))
		{
			ImGui::SeparatorText("Transform");

			if (ImGui::DragFloat3("Position", (float*)&meshInstance.position, 0.01f))
				scene.InvalidateMeshInstance(i);

			if (ImGui::DragFloat3("Rotation", (float*)&meshInstance.rotation, 0.1f))
				scene.InvalidateMeshInstance(i);

			if (ImGui::DragFloat3("Scale", (float*)&meshInstance.scale, 0.01f))
				scene.InvalidateMeshInstance(i);

			if (ImGui::TreeNode("Material"))
			{
				if (meshInstance.materialId == -1)
				{
					if (ImGui::Button("Custom material"))
						meshInstance.materialId = 0;
				}
				else
				{

					if (ImGui::Combo("Id", &meshInstance.materialId, materialsString.c_str()))
						scene.InvalidateMeshInstance(i);

					Material& material = materials[meshInstance.materialId];
					int type = (int)material.type;

					if (ImGui::Combo("Type", &type, materialTypes.c_str()))
						assetManager.InvalidateMaterial(meshInstance.materialId);

					material.type = (Material::Type)type;

					switch (material.type)
					{
					case Material::Type::LIGHT:
						if (ImGui::DragFloat3("Emission", (float*)&material.light.emission, 0.01f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					case Material::Type::DIFFUSE:
						if (ImGui::ColorEdit3("Albedo", (float*)&material.diffuse.albedo))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					case Material::Type::METAL:
						if (ImGui::ColorEdit3("Albedo", (float*)&material.diffuse.albedo))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat("Roughness", &material.plastic.roughness, 0.01f, 0.0f, 1.0f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					case Material::Type::DIELECTRIC:
						if (ImGui::DragFloat("Roughness", &material.dielectric.roughness, 0.01f, 0.0f, 1.0f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat("Refraction index", &material.dielectric.ior, 0.01f, 1.0f, 2.5f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					}
				}
				ImGui::TreePop();
				ImGui::Spacing();
			}

		}
		ImGui::PopID();
	}

	ImGui::Spacing();
	ImGui::Separator();

	ImGui::End();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport");
	
	uint32_t viewportWidth = ImGui::GetContentRegionAvail().x;
	uint32_t viewportHeight = ImGui::GetContentRegionAvail().y;

	OnResize(scene.GetCamera(), viewportWidth, viewportHeight);

	ImGui::Image((void *)(intptr_t)m_Texture->GetHandle(), ImVec2(m_Texture->GetWidth(), m_Texture->GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
	ImGui::PopStyleVar();
}

void Renderer::UpdateTimer(float deltaTime)
{
	m_NAccumulatedFrame++;
	m_AccumulatedTime += deltaTime;
	if (glfwGetTime() - m_DisplayFPSTimer >= 0.2f || m_DeltaTime == 0)
	{
		m_DisplayFPSTimer = glfwGetTime();
		m_DeltaTime = m_AccumulatedTime / m_NAccumulatedFrame;
		m_NAccumulatedFrame = 0;
		m_AccumulatedTime = 0.0f;
	}
}

void Renderer::UnpackToTexture()
{
	m_Texture->Bind();
	m_PixelBuffer->Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Texture->GetWidth(), m_Texture->GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	m_PixelBuffer->Unbind();
}

void Renderer::OnResize(std::shared_ptr<Camera> camera, uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_FrameNumber = 0;
		m_Texture->OnResize(width, height);
		m_PixelBuffer->OnResize(width, height);
		camera->OnResize(width, height);
		checkCudaErrors(cudaFree((void*)m_AccumulationBuffer));
		checkCudaErrors(cudaMalloc((void**)&m_AccumulationBuffer, width * height * sizeof(float3)));

		m_ViewportWidth = width;
		m_ViewportHeight = height;
	}
}

