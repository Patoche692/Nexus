# PathTracer

A real-time path tracer from scratch written in C++ using CUDA and OpenGL.

- [Screenshots](#screenshots)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Resources](#resources)

## Screenshots

![bathroom2](https://github.com/Patoche692/PathTracer/assets/54531293/47cddedc-bea7-43db-9d60-f91815e12622)
![dining_room2](https://github.com/Patoche692/PathTracer/assets/54531293/667db080-381b-487b-8761-befd88f85b14)
![living_room2](https://github.com/Patoche692/PathTracer/assets/54531293/c8c989b9-af19-46f2-9a47-fa36aa88ab3b)
![ellie_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/edc16123-270e-42b9-89d0-c1b843d8688d)
![classroom](https://github.com/Patoche692/PathTracer/assets/54531293/b2723f91-e41f-4ecb-b8b2-c6bc44d91dd4)
![ellie](https://github.com/Patoche692/PathTracer/assets/54531293/b180a156-0a7e-43b8-a6ed-5c01a65869b3)
![living_room3](https://github.com/Patoche692/PathTracer/assets/54531293/674132f5-035a-405a-851f-b62ae94ce0e9)
![cornell_box_spheres](https://github.com/Patoche692/PathTracer/assets/54531293/c8028e26-bb3d-45f5-bfdf-d8e1849d3c39)
<!-- ![golden_dragon_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/82be7fb0-20bd-4504-890c-4f38c00326bd) -->
![Capture d'écran 2024-04-16 203434](https://github.com/Patoche692/PathTracer/assets/54531293/a9075449-c786-4401-b6a8-28b693f81e51)
![Capture d'écran 2024-04-16 215226](https://github.com/Patoche692/PathTracer/assets/54531293/8c01378b-80f8-4bb3-b411-3a30cd65ad82)


## Prerequisites
- Having Microsoft Visual Studio installed
- Having Nvidia's [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed
- Having [Git LFS](https://git-lfs.com) installed

## Build
- Clone the repo
   ```sh
   git clone https://github.com/Patoche692/PathTracer
   ```
- The project should compile and run as is by pressing F5 in Visual Studio.

## Usage
- Controls: hold right click and use WASD keys to move and the mouse to change the camera orientation
- You can change the meshes and camera properties in the UI

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
