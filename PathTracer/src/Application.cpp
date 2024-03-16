#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_gl_interop.h>

#include "Utils.h"
#include "Renderer.cuh"

int WIDTH = 640, HEIGHT = 480;

int main(void)
{
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    window = glfwCreateWindow(WIDTH, HEIGHT, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
        std::cout << "Error initializing GLEW";

    Renderer renderer(WIDTH, HEIGHT);

    uint32_t *colors = new uint32_t[WIDTH * HEIGHT];
    std::fill_n(colors, WIDTH * HEIGHT, 0xffff0000);

    unsigned int pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uint32_t), colors, GL_DYNAMIC_DRAW);

    cudaGraphicsResource_t resource = 0;

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(1, &resource));
    size_t size = 0;
    void* device_ptr = 0;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, resource));
    
    unsigned int textureHandle;
    glGenTextures(1, &textureHandle);
    glBindTexture(GL_TEXTURE_2D, textureHandle);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	unsigned int fboId = 0;
	glGenFramebuffers(1, &fboId);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_2D, textureHandle, 0);

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        //renderer.Render(device_ptr);

        glBindTexture(GL_TEXTURE_2D, textureHandle);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);


		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  // if not already bound
        glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }


    //checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, 0));

    //glDeleteBuffers(1, &pbo
    glfwTerminate();
    return 0;
}