#include "Renderer.cuh"

//#include <vector>
//#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//
//#include "Utils.h"
//
__global__ void traceRay(void *device_ptr, uint32_t imageWidth, uint32_t imageHeight)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float x = (float)i / (float)imageWidth * 2.0f - 1.0f;
	float y = (float)j / (float)imageHeight * 2.0f - 1.0f;

	if (i >= imageWidth || j >= imageHeight)
		return;

	uint32_t* imagePtr = (uint32_t*)device_ptr;

	glm::vec3 rayOrigin(0, 0, 2.0f);
	glm::vec3 rayDirection(x, y, -1.0f);

	float radius = 0.5f;

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;


	float discriminant = b * b - 4.0f * a * c;

	if (discriminant < 0.0f)
	{
		imagePtr[j * imageWidth + i] = 0xff000000;
		return;
	}

	float t0 = (- b + glm::sqrt(discriminant)) / 2.0f * a;
	float t1 = (- b - glm::sqrt(discriminant)) / 2.0f * a;

	glm::vec3 hitPoint = rayOrigin + rayDirection * t1;
	glm::vec3 normal = glm::normalize(hitPoint);

	glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));

	float d = glm::max(glm::dot(normal, -lightDir), 0.0f);

	glm::vec3 sphereColor(1.0f, 0.0f, 1.0f);
	sphereColor = sphereColor * d;

	glm::vec4 color = glm::clamp(glm::vec4(sphereColor, 1.0f), glm::vec4(0.0f), glm::vec4(1.0f));
	uint8_t red = (uint8_t)(color.r * 255.0f);
	uint8_t green = (uint8_t)(color.g * 255.0f);
	uint8_t blue = (uint8_t)(color.b * 255.0f);
	uint8_t alpha = (uint8_t)(color.a * 255.0f);
	 
	imagePtr[j * imageWidth + i] = alpha << 24 | blue << 16 | green << 8 | red;

}

void Renderer::Render(void *device_ptr)
{ 
	uint32_t tx = 8, ty = 8;
	dim3 blocks(m_ImageWidth / tx + 1, m_ImageHeight / ty + 1);
	dim3 threads(tx, ty);

	traceRay<<<blocks, threads>>>(device_ptr, m_ImageWidth, m_ImageHeight);
}
//
//
//void Renderer::Render(const Camera& camera)
//{
//	Ray ray;
//	ray.Origin = camera.GetPosition();
//	m_FinalImage->SetData(m_ImageData);
//	//int device = -1;
//	//checkCudaErrors(cudaGetDevice(&device));
//	//checkCudaErrors(cudaMemPrefetchAsync(m_ImageData, m_FinalImage->GetWidth() * m_FinalImage->GetHeight() * sizeof(uint32_t), device, NULL));
//
//	PerPixel(m_ImageData, m_FinalImage->GetWidth(), m_FinalImage->GetHeight());
//	checkCudaErrors(cudaGetLastError());
//	//for (uint32_t y = 0; y < m_FinalImage->GetHeight(); y++)
//	//{
//	//	for (uint32_t x = 0; x < m_FinalImage->GetWidth(); x++)
//	//	{
//	//		ray.Direction = camera.GetRayDirections()[y * m_FinalImage->GetWidth() + x];
//
//
//	//		color = glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f));
//	//		m_ImageData[y * m_FinalImage->GetWidth() + x] = Utils::ConvertToRGBA(color);
//	//	}
//	//}
//	checkCudaErrors(cudaDeviceSynchronize());
//}
//
//glm::vec4 Renderer::TraceRay(const Ray& ray)
//{
//	float radius = 0.5f;
//
//	float a = glm::dot(ray.Direction, ray.Direction);
//	float b = 2.0f * glm::dot(ray.Origin, ray.Direction);
//	float c = glm::dot(ray.Origin, ray.Origin) - radius * radius;
//
//
//	float discriminant = b * b - 4.0f * a * c;
//
//	if (discriminant < 0.0f)
//	{
//		return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
//	}
//
//	float t0 = (- b + glm::sqrt(discriminant)) / 2.0f * a;
//	float t1 = (- b - glm::sqrt(discriminant)) / 2.0f * a;
//
//	glm::vec3 hitPoint = ray.Origin + ray.Direction * t1;
//	glm::vec3 normal = glm::normalize(hitPoint);
//
//	glm::vec3 lightDir = glm::normalize(glm::vec3(-1.0f, -1.0f, -1.0f));
//
//	float d = glm::max(glm::dot(normal, -lightDir), 0.0f);
//
//	glm::vec3 sphereColor(1.0f, 0.0f, 1.0f);
//	sphereColor = sphereColor * d;
//
//	return glm::vec4(sphereColor, 1.0f);
//}
//
//
//void Renderer::OnResize(uint32_t width, uint32_t height)
//{
//	if (m_FinalImage)
//	{
//		// No resize necessary
//		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
//			return;
//
//		m_FinalImage->Resize(width, height);
//	}
//	else 
//	{
//		m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
//	}
//
//	//delete[] m_ImageData;
//	//m_ImageData = new uint32_t[width * height];
//	checkCudaErrors(cudaFree(m_ImageData));
//	checkCudaErrors(cudaMallocManaged((void**)&m_ImageData, width * height * sizeof(uint32_t)));
//}
