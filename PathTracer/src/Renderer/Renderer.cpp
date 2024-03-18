#include "Renderer.h"
#include "Renderer.cuh"
#include "../Utils.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


Renderer::Renderer(uint32_t width, uint32_t height, GLFWwindow* window)
	:m_ImageWidth(width), m_ImageHeight(height)
{
	m_TextureRenderer = std::make_shared<TextureRenderer>(width, height);
	m_UIRenderer = std::make_shared<UIRenderer>(window, m_TextureRenderer->GetTexture()->GetHandle(), width, height);
}

void Renderer::Render()
{ 
	std::shared_ptr<PixelBuffer> pixelBuffer = m_TextureRenderer->GetPixelBuffer();
	std::shared_ptr<Texture> texture = m_TextureRenderer->GetTexture();

	m_UIRenderer->Render(texture, pixelBuffer);

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

void Renderer::OnResize(uint32_t width, uint32_t height)
{
}

