#pragma once

#include <glm.hpp>
#include <cuda_runtime_api.h>
#include "UIRenderer.h"
#include "../OpenGL_API/TextureRenderer.h"
#include <GLFW/glfw3.h>

//#include "Camera.h"
//#include "Ray.h"
//
class Renderer
{
public:
	Renderer(uint32_t width, uint32_t height, GLFWwindow* window);

	void OnResize(uint32_t width, uint32_t height);
	void Render();

	//std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }

private:
	//glm::vec4 TraceRay(const Ray& ray);
private:
	//std::shared_ptr<Walnut::Image> m_FinalImage;
	//uint32_t* m_ImageData = nullptr;
	uint32_t m_ImageWidth, m_ImageHeight;
	std::shared_ptr<TextureRenderer> m_TextureRenderer;
	std::shared_ptr<UIRenderer> m_UIRenderer;
};

