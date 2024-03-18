#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "RayTracerApplication.h"

int WIDTH = 1200, HEIGHT = 800;

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

    // Disable vsync (frame rate / screen refresh rate synchronization)
    //glfwSwapInterval(0);

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
