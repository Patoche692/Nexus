# PathTracer

A real-time path tracer from scratch written in C++ using CUDA and OpenGL.

- [Screenshots](#screenshots)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Resources](#resources)

## Screenshots

![cannelle_et_fromage2](https://github.com/Patoche692/PathTracer/assets/54531293/9a2f440f-2556-4e9c-9d65-308bf2416424)
![sormtrooper2](https://github.com/Patoche692/PathTracer/assets/54531293/2de0eb11-19db-48f7-860c-96747e73c734)
![monster_under_bed](https://github.com/Patoche692/PathTracer/assets/54531293/fdd2a636-e2ef-47cf-8449-c7b2c030d534)
![piano3](https://github.com/Patoche692/PathTracer/assets/54531293/905c2bce-2aac-4b43-818e-ff928d16aab4)
![piano_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/138c3838-6097-49fd-a905-b48878f885d9)
![lamp](https://github.com/Patoche692/PathTracer/assets/54531293/d8344999-7289-43be-bf91-b9e99ff67e7d)
![rolls_royce](https://github.com/Patoche692/PathTracer/assets/54531293/9af03cd7-273b-4bad-bf69-3a73ff2f6604)
![rolls_royce4](https://github.com/Patoche692/PathTracer/assets/54531293/244558e1-872b-45f5-ac1f-b6b38f027ba0)
![coffee](https://github.com/Patoche692/PathTracer/assets/54531293/b860d5a9-99b1-43ef-ad98-8ae17d41a931)
![iron_man](https://github.com/Patoche692/PathTracer/assets/54531293/700463ed-03cc-412c-a283-ac726a1282ef)
![junk_shop](https://github.com/Patoche692/PathTracer/assets/54531293/1c46544b-8889-4b02-bd82-86924ffc36b3)
<!--![harry_potter](https://github.com/Patoche692/PathTracer/assets/54531293/eaf5fdb6-845c-4a3d-9dde-4c61c1286af2) -->
![bathroom4](https://github.com/Patoche692/PathTracer/assets/54531293/56532d4a-ebc6-4102-9830-bd66089fcd19)
<!--![bathroom2](https://github.com/Patoche692/PathTracer/assets/54531293/47cddedc-bea7-43db-9d60-f91815e12622) -->
<!--![dining_room2](https://github.com/Patoche692/PathTracer/assets/54531293/667db080-381b-487b-8761-befd88f85b14) -->
<!--![living_room2](https://github.com/Patoche692/PathTracer/assets/54531293/c8c989b9-af19-46f2-9a47-fa36aa88ab3b) -->
<!--![ellie_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/edc16123-270e-42b9-89d0-c1b843d8688d) -->
![cornell_box_spheres](https://github.com/Patoche692/PathTracer/assets/54531293/c8028e26-bb3d-45f5-bfdf-d8e1849d3c39)
<!-- ![golden_dragon_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/82be7fb0-20bd-4504-890c-4f38c00326bd) -->
<!--![Capture d'écran 2024-04-16 203434](https://github.com/Patoche692/PathTracer/assets/54531293/a9075449-c786-4401-b6a8-28b693f81e51) -->
<!--![Capture d'écran 2024-04-16 215226](https://github.com/Patoche692/PathTracer/assets/54531293/8c01378b-80f8-4bb3-b411-3a30cd65ad82) -->


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
