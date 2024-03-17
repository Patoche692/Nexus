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
        std::cout << "Error creating glfw window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK)
        std::cout << "Error initializing GLEW" << std::endl;

    RayTracerApplication rayTracerApplication(WIDTH, HEIGHT, window);

    int width, height;
    double startTime, elapsedTime;
    startTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        glClear(GL_COLOR_BUFFER_BIT);

        glfwGetWindowSize(window, &width, &height);
        rayTracerApplication.OnResize(width, height);

        elapsedTime = glfwGetTime() - startTime;
        startTime = glfwGetTime();
        rayTracerApplication.Update(elapsedTime);

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
