#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cuda_gl_interop.h>
#include "RayTracerApplication.h"

int WIDTH = 640, HEIGHT = 640;

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

    RayTracerApplication rayTracerApplication(WIDTH, HEIGHT);

    double startTime, elapsedTime;
    startTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);

        elapsedTime = glfwGetTime() - startTime;
        startTime = glfwGetTime();
        rayTracerApplication.Update(elapsedTime);

        glfwSwapBuffers(window);

        glfwPollEvents();
        //std::cout << elapsedTime * 1000 << std::endl;
    }

    glfwTerminate();
    return 0;
}
