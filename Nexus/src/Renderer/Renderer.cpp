#include <glm/gtc/type_ptr.hpp>
#include "Renderer.h"
#include "Cuda/PathTracer.cuh"
#include "Utils/Utils.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "FileDialog.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "windows.h"
#include <string>
#include <iostream>
#include <direct.h>

Renderer::Renderer(uint32_t width, uint32_t height, GLFWwindow* window)
	:m_ViewportWidth(width), m_ViewportHeight(height)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	TCHAR NPath[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, NPath);
	std::cout << NPath;
	io.FontDefault = io.Fonts->AddFontFromFileTTF("assets/fonts/opensans/OpenSans-Regular.ttf", 16.0f);
    ImGui::StyleColorsCustomDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");

	m_PixelBuffer = std::make_shared<PixelBuffer>(width, height);
	m_Texture = std::make_shared<OGLTexture>(width, height);

	checkCudaErrors(cudaMalloc((void**)&m_AccumulationBuffer, width * height * sizeof(float3)));

	m_DisplayFPSTimer = glfwGetTime();

	GLFWwindow* glfwWindow = window; 

}

Renderer::~Renderer()
{
	checkCudaErrors(cudaFree((void*)m_AccumulationBuffer));
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Renderer::Reset()
{
	m_FrameNumber = 0;
	m_MRPS = 0;
	m_NumRaysProcessed = 0;
	m_PixelBuffer = std::make_shared<PixelBuffer>(m_ViewportWidth, m_ViewportHeight);
	checkCudaErrors(cudaMalloc((void**)&m_AccumulationBuffer, m_ViewportWidth * m_ViewportHeight * sizeof(float3)));
	m_DisplayFPSTimer = glfwGetTime();
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

	// Launch cuda path tracing kernel, writes the viewport into the pixelbuffer
	if (!scene.IsEmpty())
	{
		m_FrameNumber++;

		RenderViewport(m_PixelBuffer, m_FrameNumber, m_AccumulationBuffer);

		// Unpack the pixel buffer written by cuda to the renderer texture
		UnpackToTexture();

	}
	else
		m_FrameNumber = 0;


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

static bool DrawFloat3Control(const std::string& label, float3& values, float resetValue = 0.0f, float columnWidth = 80.0f)
{
	ImGui::PushID(label.c_str());

	ImGui::Columns(2);
	ImGui::SetColumnWidth(0, columnWidth);
	ImGui::Text(label.c_str());
	ImGui::NextColumn();

	ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

	float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
	ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

	bool modified = false;

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f });
	if (ImGui::Button("X", buttonSize))
		values.x = resetValue, modified = true;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f"))
		modified = true;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3f, 0.8f, 0.3f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2f, 0.7f, 0.2f, 1.0f });
	if (ImGui::Button("Y", buttonSize))
		values.y = resetValue, modified = true;;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f"))
		modified = true;
	ImGui::PopItemWidth();
	ImGui::SameLine();

	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f });
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f });
	if (ImGui::Button("Z", buttonSize))
		values.z = resetValue, modified = true;
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	if (ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f"))
		modified = true;
	ImGui::PopItemWidth();

	ImGui::PopStyleVar();

	ImGui::Columns(1);

	ImGui::PopID();

	return modified;
}

void Renderer::RenderUI(Scene& scene)
{
	ImGui::DockSpaceOverViewport();

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open...", "Ctrl+O"))
			{
				std::string fullPath = FileDialog::OpenFile(
					"3D model (*.obj;*.ply;*.stl;*.glb;*.gltf;*.fbx;*.3ds;*.blend;*.dae)\0*.obj;*.ply;*.stl;*.glb;*.gltf;*.fbx;*.3ds;*.blend;*.dae\0"
				);
				if (!fullPath.empty())
				{
					checkCudaErrors(cudaDeviceSynchronize());
					checkCudaErrors(cudaDeviceReset());
					Reset();
					scene.Reset();

					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					scene.CreateMeshInstanceFromFile(filePath, fileName);
					checkCudaErrors(cudaDeviceSynchronize());
				}
			}

			if (ImGui::MenuItem("Load HDR map", "Ctrl+H"))
			{
				std::string fullPath = FileDialog::OpenFile(
					"HDR file (*.hdr)\0*.hdr;*.exr\0"
				);
				if (!fullPath.empty())
				{
					std::string fileName, filePath;
					Utils::GetPathAndFileName(fullPath, filePath, fileName);
					scene.AddHDRMap(filePath, fileName);
					m_FrameNumber = 0;
				}
			}

			if (ImGui::MenuItem("Save Screenshot", "Ctrl+S")) {
				SaveScreenshot("screenshot.png");
			}

			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	ImGui::Begin("Settings");

	ImGui::Spacing();
	ImGui::Separator();
	ImGui::Text("Time info");
	ImGui::Text("Render time millisec: %.3f", m_DeltaTime);
	ImGui::Text("FPS: %d", (int)(1000.0f / m_DeltaTime));
	ImGui::Text("Frame: %d", m_FrameNumber);
	ImGui::Text("Megarays/sec: %.2f", m_MRPS);

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

		if (ImGui::CollapsingHeader(meshInstance.name.c_str()))
		{
			ImGui::SeparatorText("Transform");

			if (DrawFloat3Control("Location", meshInstance.position))
				scene.InvalidateMeshInstance(i);

			if (DrawFloat3Control("Rotation", meshInstance.rotation))
				scene.InvalidateMeshInstance(i);

			if (DrawFloat3Control("Scale", meshInstance.scale, 1.0f))
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
					case Material::Type::DIFFUSE:
						if (ImGui::ColorEdit3("Albedo", (float*)&material.diffuse.albedo))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					case Material::Type::DIELECTRIC:
						if (ImGui::ColorEdit3("Albedo", (float*)&material.dielectric.albedo))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat("Roughness", &material.dielectric.roughness, 0.01f, 0.0f, 1.0f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat("Transmittance", &material.dielectric.transmittance, 0.01f, 0.0f, 1.0f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat("Refraction index", &material.dielectric.ior, 0.01f, 1.0f, 2.5f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					case Material::Type::CONDUCTOR:
						if (ImGui::DragFloat("Roughness", &material.conductor.roughness, 0.01f, 0.0f, 1.0f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat3("Refraction index", (float*)&material.conductor.ior, 0.01f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						if (ImGui::DragFloat3("k", (float*)&material.conductor.k, 0.01f))
							assetManager.InvalidateMaterial(meshInstance.materialId);
						break;
					}
					if (ImGui::DragFloat3("Emission", (float*)&material.emissive, 0.01f))
						assetManager.InvalidateMaterial(meshInstance.materialId);
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

	scene.GetCamera()->OnResize(viewportWidth, viewportHeight);
	OnResize(viewportWidth, viewportHeight);

	ImGui::Image((void *)(intptr_t)m_Texture->GetHandle(), ImVec2(m_Texture->GetWidth(), m_Texture->GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();

	ImGui::PopStyleVar();
}

void Renderer::UpdateTimer(float deltaTime)
{
	m_NAccumulatedFrame++;
	m_NumRaysProcessed += m_ViewportHeight * m_ViewportWidth;

	m_AccumulatedTime += deltaTime;
	if (glfwGetTime() - m_DisplayFPSTimer >= 0.2f || m_DeltaTime == 0)
	{
		m_DisplayFPSTimer = glfwGetTime();
		m_DeltaTime = m_AccumulatedTime / m_NAccumulatedFrame;
		m_MRPS = static_cast<float>(m_NumRaysProcessed) / m_AccumulatedTime / 1000.0f;		// millisecond * 1.000.000
		
		m_NAccumulatedFrame = 0;
		m_AccumulatedTime = 0.0f;
		m_NumRaysProcessed = 0;
	}
}

void Renderer::UnpackToTexture()
{
	m_Texture->Bind();
	m_PixelBuffer->Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Texture->GetWidth(), m_Texture->GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	m_PixelBuffer->Unbind();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_FrameNumber = 0;
		m_MRPS = 0;
		m_NumRaysProcessed = 0;
		m_Texture->OnResize(width, height);
		m_PixelBuffer->OnResize(width, height);
		checkCudaErrors(cudaFree((void*)m_AccumulationBuffer));
		checkCudaErrors(cudaMalloc((void**)&m_AccumulationBuffer, width * height * sizeof(float3)));

		m_ViewportWidth = width;
		m_ViewportHeight = height;
	}
}

void Renderer::SaveScreenshot(const std::string& filename)
{
	char buffer[FILENAME_MAX];
	std::string cwd = _getcwd(buffer, FILENAME_MAX);
	std::string filepath = cwd + "\\" + filename;

	int width = m_Texture->GetWidth();
	int height = m_Texture->GetHeight();
	std::vector<unsigned char> pixels(width * height * 4);

	glBindTexture(GL_TEXTURE_2D, m_Texture->GetHandle());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
	glBindTexture(GL_TEXTURE_2D, 0);

	stbi_flip_vertically_on_write(1);
	if (!stbi_write_png(filepath.c_str(), width, height, 4, pixels.data(), width * 4))
	{
		std::cerr << "Failed to save screenshot to " << filepath << std::endl;
	}

	std::cout << "Screenshot saved at: " << filepath.c_str() << std::endl;
}
