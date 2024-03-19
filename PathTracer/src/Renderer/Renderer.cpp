#include "Renderer.h"
#include "Renderer.cuh"
#include "../Utils.h"
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
	m_Texture = std::make_shared<Texture>(width, height);
}

Renderer::~Renderer()
{

	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


void Renderer::Render(Camera* camera, float deltaTime)
{ 
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	if (camera->HasMoved())
		SendCameraDataToDevice(camera);
	// Position UI and resize the texture and pixel buffer depending on the viewport size
	RenderUI(camera, deltaTime);

	// Launch cuda path tracing kernel, writes the viewport into the pixelbuffer
	RenderViewport(m_PixelBuffer);

	// Unpack the pixel buffer written by cuda to the renderer texture
	UnpackToTexture();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::RenderUI(Camera* camera, float deltaTime)
{
	ImGui::DockSpaceOverViewport();

	ImGui::Begin("Settings");
	ImGui::Text("Render time millisec: %.3f", deltaTime);
	ImGui::Text("FPS: %d", (int)(1000.0f / deltaTime));
	ImGui::End();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport");
	
	uint32_t viewportWidth = ImGui::GetContentRegionAvail().x;
	uint32_t viewportHeight = ImGui::GetContentRegionAvail().y;

	OnResize(camera, viewportWidth, viewportHeight);

	ImGui::Image((void *)(intptr_t)m_Texture->GetHandle(), ImVec2(m_Texture->GetWidth(), m_Texture->GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
	ImGui::PopStyleVar();
}

void Renderer::UnpackToTexture()
{
	m_Texture->Bind();
	m_PixelBuffer->Bind();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Texture->GetWidth(), m_Texture->GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	m_PixelBuffer->Unbind();
}

void Renderer::OnResize(Camera* camera, uint32_t width, uint32_t height)
{
	if ((m_ViewportWidth != width || m_ViewportHeight != height) && width != 0 && height != 0)
	{
		m_Texture->OnResize(width, height);
		m_PixelBuffer->OnResize(width, height);
		camera->OnResize(width, height);

		m_ViewportWidth = width;
		m_ViewportHeight = height;
	}
}

