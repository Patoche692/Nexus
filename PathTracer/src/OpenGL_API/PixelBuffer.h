#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

class PixelBuffer
{
public:
	PixelBuffer(uint32_t width, uint32_t height);
	~PixelBuffer();

	void Bind();
	void Unbind();
	void OnResize(uint32_t width, uint32_t height);

	unsigned int GetHandle() { return m_Handle; };
	cudaGraphicsResource_t& GetCudaResource() { return m_CudaResource; };

private:
	unsigned int m_Handle = 0;
	cudaGraphicsResource_t m_CudaResource;
};
