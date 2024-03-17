#include "UIRenderer.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

UIRenderer::UIRenderer(GLFWwindow *window, unsigned int viewportTextureHandle, uint32_t viewportWidth, uint32_t viewportHeight)
	:m_ViewportTextureHandle(viewportTextureHandle), m_ViewportWidth(viewportWidth), m_ViewportHeight(viewportHeight)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");
}

UIRenderer::~UIRenderer()
{
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UIRenderer::Render(std::shared_ptr<Texture> texture, std::shared_ptr<PixelBuffer> pixelBuffer)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::DockSpaceOverViewport();

	ImGui::Begin("Settings");
	ImGui::Text("Render time millisec:");
	ImGui::End();

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("Viewport");
	
	uint32_t viewportWidth = ImGui::GetContentRegionAvail().x;
	uint32_t viewportHeight = ImGui::GetContentRegionAvail().y;
	if (m_ViewportWidth != viewportWidth || m_ViewportHeight != viewportHeight)
	{
		texture->OnResize(viewportWidth, viewportHeight);
		pixelBuffer->OnResize(viewportWidth, viewportHeight);

		m_ViewportWidth = ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = ImGui::GetContentRegionAvail().y;
	}

	ImGui::Image((void *)(intptr_t)texture->GetHandle(), ImVec2(texture->GetWidth(), texture->GetHeight()), ImVec2(0, 1), ImVec2(1, 0));

	ImGui::End();
	ImGui::PopStyleVar();


}
