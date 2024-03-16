#pragma once

//#include <memory>
#include <glm.hpp>
#include <cuda_runtime_api.h>

//#include "Camera.h"
//#include "Ray.h"
//
class Renderer
{
public:
	Renderer() = default;
	Renderer(uint32_t width, uint32_t height)
		:m_ImageWidth(width), m_ImageHeight(height) {}

	//void OnResize(uint32_t width, uint32_t height);
	void Render(void* device_ptr);

	//std::shared_ptr<Walnut::Image> GetFinalImage() const { return m_FinalImage; }

private:
	//glm::vec4 TraceRay(const Ray& ray);
private:
	//std::shared_ptr<Walnut::Image> m_FinalImage;
	//uint32_t* m_ImageData = nullptr;
	uint32_t m_ImageWidth, m_ImageHeight;

};

