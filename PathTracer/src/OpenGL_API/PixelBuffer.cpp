#include "PixelBuffer.h"
#include "../Utils.h"

PixelBuffer::PixelBuffer(uint32_t width, uint32_t height)
{
    glGenBuffers(1, &m_Handle);
    Bind();
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uint32_t), NULL, GL_DYNAMIC_DRAW);

    // Register the buffer for CUDA to use
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_CudaResource, m_Handle, cudaGraphicsRegisterFlagsWriteDiscard));
}

void PixelBuffer::Bind()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Handle);
}

PixelBuffer::~PixelBuffer()
{
    glDeleteBuffers(1, &m_Handle);
}
