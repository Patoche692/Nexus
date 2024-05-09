# PathTracer

Interactive physically based GPU path tracer from scratch written in C++ using CUDA and OpenGL.

- [Screenshots](#screenshots)
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Resources](#resources)

## Screenshots

![cannelle_et_fromage3](https://github.com/Patoche692/PathTracer/assets/54531293/1356478c-4c1c-4192-93fb-3798a642b5f4)
![sormtrooper2](https://github.com/Patoche692/PathTracer/assets/54531293/2de0eb11-19db-48f7-860c-96747e73c734)
![mustang](https://github.com/Patoche692/PathTracer/assets/54531293/ffa3f777-da30-4935-92d9-2c21f2d0bc0b)
![monster_under_bed](https://github.com/Patoche692/PathTracer/assets/54531293/fdd2a636-e2ef-47cf-8449-c7b2c030d534)
![bathroom5](https://github.com/Patoche692/PathTracer/assets/54531293/d3a828f9-3cb1-4bf7-abce-e193a9968538)
![piano3](https://github.com/Patoche692/PathTracer/assets/54531293/905c2bce-2aac-4b43-818e-ff928d16aab4)
![piano_zoom](https://github.com/Patoche692/PathTracer/assets/54531293/138c3838-6097-49fd-a905-b48878f885d9)
![lamp](https://github.com/Patoche692/PathTracer/assets/54531293/d8344999-7289-43be-bf91-b9e99ff67e7d)
<!--![rolls_royce](https://github.com/Patoche692/PathTracer/assets/54531293/9af03cd7-273b-4bad-bf69-3a73ff2f6604)-->
![rolls_royce4](https://github.com/Patoche692/PathTracer/assets/54531293/244558e1-872b-45f5-ac1f-b6b38f027ba0)
![coffee](https://github.com/Patoche692/PathTracer/assets/54531293/b860d5a9-99b1-43ef-ad98-8ae17d41a931)
<!--![iron_man](https://github.com/Patoche692/PathTracer/assets/54531293/700463ed-03cc-412c-a283-ac726a1282ef)-->
![junk_shop](https://github.com/Patoche692/PathTracer/assets/54531293/1c46544b-8889-4b02-bd82-86924ffc36b3)
<!--![cornell_box_spheres](https://github.com/Patoche692/PathTracer/assets/54531293/c8028e26-bb3d-45f5-bfdf-d8e1849d3c39)-->


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

## Models
- [LuxCore example scenes](https://luxcorerender.org/example-scenes/)
- [Blender demo scenes](https://www.blender.org/download/demo-files/)
- [Stormtrooper](https://www.blendswap.com/blend/13953) by [ScottGraham](https://www.blendswap.com/profile/120125)
- [Ford mustang](https://sketchfab.com/3d-models/ford-mustang-1965-5f4e3965f79540a9888b5d05acea5943) by [Pooya_dh](https://sketchfab.com/Pooya_dh)
- [Bathroom](https://www.blendswap.com/blend/12584) by [nacimus](https://www.blendswap.com/profile/72536)
- [Piano](https://blendswap.com/blend/29080) by [Roy](https://blendswap.com/profile/1508348)
- [Lamp](https://www.blendswap.com/blend/6885) by [UP3D](https://www.blendswap.com/profile/4758)
- [Rolls Royce](https://www.blenderkit.com/asset-gallery-detail/3654527a-4b8d-4392-a863-515276fbf541/) by [Jayrenn Reeve](https://www.blenderkit.com/asset-gallery?query=author_id:114910)
- [Coffee Maker](https://blendswap.com/blend/16368) by [cekuhnen](https://blendswap.com/profile/13522)
