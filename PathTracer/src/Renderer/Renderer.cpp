#include "Renderer.h"
#include "Renderer.cuh"
#include "../Utils.h"


Renderer::Renderer(uint32_t width, uint32_t height, GLFWwindow* window)
	:m_ImageWidth(width), m_ImageHeight(height)
{
	m_TextureRenderer = std::make_shared<TextureRenderer>(width, height);
	m_UIRenderer = std::make_shared<UIRenderer>(window, m_TextureRenderer->GetTexture()->GetHandle(), width, height);
}

void Renderer::Render()
{ 
	std::shared_ptr<PixelBuffer> pixelBuffer = m_TextureRenderer->GetPixelBuffer();
	checkCudaErrors(cudaGraphicsMapResources(1, &pixelBuffer->GetCudaResource()));
	size_t size = 0;
	void* device_ptr = 0;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, pixelBuffer->GetCudaResource()));

	// Launch cuda path tracing kernel
	cudaRender(device_ptr, m_ImageWidth, m_ImageHeight);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &pixelBuffer->GetCudaResource(), 0));

	m_TextureRenderer->Render();

	m_UIRenderer->Render(m_TextureRenderer->GetTexture());

	// Display on screen the texture rendered by the TextureRenderer
	//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	//glBlitFramebuffer(0, 0, m_ImageWidth, m_ImageHeight, 0, 0, m_ImageWidth, m_ImageHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_ImageWidth == width && m_ImageHeight == height)
	{
		return;
	}
}

