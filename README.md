# PathTracer

A real-time path tracer from scratch written in C++ using CUDA and OpenGL.

- [Screenshots](#screenshots)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Dependencies](#dependencies)
- [Resources](#resources)

## Screenshots

![cornell_box_spheres](https://github.com/Patoche692/PathTracer/assets/54531293/c8028e26-bb3d-45f5-bfdf-d8e1849d3c39)
![golden_dragon_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/82be7fb0-20bd-4504-890c-4f38c00326bd)
![glass_dragon](https://github.com/Patoche692/PathTracer/assets/54531293/49065a85-b623-40d9-a3f9-e5a4af1625ac)
![Capture d'écran 2024-04-13 0017481](https://github.com/Patoche692/PathTracer/assets/54531293/d64676fb-3d1d-4a99-b031-bf4b4bb252c4)
![Capture d'écran 2024-04-06 194314](https://github.com/Patoche692/PathTracer/assets/54531293/4b5813a6-61ea-45bc-a2c7-5e110d82e9d8)

## Prerequisites
- Having Microsoft Visual Studio installed
- Having Nvidia's [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed

## Build
- Clone the repo
   ```sh
   git clone https://github.com/Patoche692/PathTracer
   ```
- The project should compile and run as is by pressing F5.

## Dependencies
- [GLFW](https://www.glfw.org) and [GLEW](https://glew.sourceforge.net)
- [glm](https://github.com/g-truc/glm)
- [CUDA](https://developer.nvidia.com/cuda-downloads) 12.4
- [CUDA math helper](https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h) for common operations on CUDA vector types
- [Assimp](https://github.com/assimp/assimp) for model loading
- [ImGui](https://github.com/ocornut/imgui) for user interface


## Resources
- [Ray Tracing in one weekend series](https://raytracing.github.io)
- [Accelerated Ray Tracing in one weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- [Physically based rendering book](https://www.pbr-book.org/4ed/contents)
- [Ray Tracing Gems II: Next Generation Real-Time Rendering with DXR, Vulkan, and OptiX](https://www.realtimerendering.com/raytracinggems/rtg2/index.html)
- The Cherno's [Ray tracing series](https://www.youtube.com/playlist?list=PLlrATfBNZ98edc5GshdBtREv5asFW3yXl)
- [ScratchPixel website](https://scratchapixel.com)
- Jacco Bikker's [guides](https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/) on fast SAH-based BVH construction
