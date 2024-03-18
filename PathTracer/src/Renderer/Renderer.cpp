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

	m_TextureRenderer = std::make_shared<TextureRenderer>(width, height);
}

Renderer::~Renderer()
{

	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


void Renderer::Render(float deltaTime)
{ 
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// Position UI and resize the texture and pixel buffer depending on the viewport size
	RenderUI(deltaTime);

	std::shared_ptr<PixelBuffer> pixelBuffer = m_TextureRenderer->GetPixelBuffer();
	std::shared_ptr<Texture> texture = m_TextureRenderer->GetTexture();

	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	void* device_ptr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, pixelBuffer->GetCudaResource()));

	// Launch cuda path tracing kernel
	cudaRender(device_ptr, texture->GetWidth(), texture->GetHeight());

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));

	m_TextureRenderer->Render();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Display on screen the texture rendered by the TextureRenderer
	//std::shared_ptr<Framebuffer> framebuffer = m_TextureRenderer->GetFramebuffer();
	//framebuffer->AttachToTextureHandle(texture->GetHandle());

	//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	//glBlitFramebuffer(0, 0, 800, 800, 0, 0, 800, 800, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Renderer::RenderUI(float deltaTime)
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

	OnResize(viewportWidth, viewportHeight);

	std::shared_ptr<Texture> texture = m_TextureRenderer->GetTexture();
	ImGui::Image((void *)(intptr_t)texture->GetHandle(), ImVec2(texture->GetWidth(), texture->GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
	ImGui::PopStyleVar();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_ViewportWidth != width || m_ViewportHeight != height)
	{
		std::shared_ptr<PixelBuffer> pixelBuffer = m_TextureRenderer->GetPixelBuffer();
		std::shared_ptr<Texture> texture = m_TextureRenderer->GetTexture();

		texture->OnResize(width, height);
		pixelBuffer->OnResize(width, height);

		m_ViewportWidth = width;
		m_ViewportHeight = height;
	}
}

