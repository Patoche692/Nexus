#pragma once
#include <iostream>

#include "Framebuffer.h"
#include "PixelBuffer.h"
#include "Texture.h"

/**
 * This class is used to unpack the pixel buffer (updated every frame by cuda)
 * to the opengl texture which is linked to the framebuffer and can finally be rendered on the screen.
 *
 * Every operation (pixel buffer write with cuda, unpacking to texture) happens gpu side for better performance.
 */
class TextureRenderer
{
public:
	TextureRenderer(uint32_t width, uint32_t height);

	void Render();

	std::shared_ptr<PixelBuffer> GetPixelBuffer() { return m_PixelBuffer; };
	std::shared_ptr<Texture> GetTexture() { return m_Texture; };
	std::shared_ptr<Framebuffer> GetFramebuffer() { return m_Framebuffer; };

private:
	std::shared_ptr<PixelBuffer> m_PixelBuffer;
	std::shared_ptr<Texture> m_Texture;
	std::shared_ptr<Framebuffer> m_Framebuffer;
	uint32_t m_Width, m_Height;
};
