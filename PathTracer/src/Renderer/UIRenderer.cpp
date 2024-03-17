#include "UIRenderer.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

UIRenderer::UIRenderer(GLFWwindow *window, unsigned int viewportTextureHandle, uint32_t viewportTextureWidth, uint32_t viewportTextureHeight)
	:m_ViewportTextureHandle(viewportTextureHandle), m_ViewportTextureWidth(viewportTextureWidth), m_ViewportTextureHeight(viewportTextureHeight)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");
}

UIRenderer::~UIRenderer()
{
	ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UIRenderer::Render(std::shared_ptr<Texture> texture)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Settings");
	ImGui::Text("Render time millisec:");
	ImGui::End();

	ImGui::Begin("Viewport");
	ImGui::Image((void *)(intptr_t)texture->GetHandle(), ImVec2(texture->GetWidth(), texture->GetHeight()));
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
