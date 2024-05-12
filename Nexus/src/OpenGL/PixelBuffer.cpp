#include "PixelBuffer.h"
#include "../Utils/Utils.h"

PixelBuffer::PixelBuffer(uint32_t width, uint32_t height)
    :m_Width(width), m_Height(height)
{
    glGenBuffers(1, &m_Handle);
    Bind();
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);

    // Register the buffer for CUDA to use
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Handle, cudaGraphicsRegisterFlagsWriteDiscard));
    Unbind();
}

void PixelBuffer::Bind()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Handle);
}

void PixelBuffer::Unbind()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void PixelBuffer::OnResize(uint32_t width, uint32_t height)
{
    m_Width = width;
    m_Height = height;
    Bind();
    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaResource));
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Handle, cudaGraphicsRegisterFlagsWriteDiscard));
    Unbind();
}

PixelBuffer::~PixelBuffer()
{
    //checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaResource));
    glDeleteBuffers(1, &m_Handle);
}
