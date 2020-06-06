# CUDA_Ray_Tracer
Partial CUDA accelerated implementation of the software ray tracer students have to write during the Graphics Programming 1 course.
All source is written by myself. The ray tracer is still **"Under Development"** and will be extended with optimization techniques, transparency, triangle mesh supporty, etc. Next to the topics covered in class, this version also supports **progressive indirect lighting** using **importance sampling** based on the BRDF's.

A **Nvidia GPU** is **required** to run the demo. If you want to run the Visual Studio project, please use **Visual Studio 2019** and have the **CUDA SDK** installed.

## Interesting files
- [Tracer](_PROJECT/source/Raytracer/Tracer.cu): the heart of the ray tracer, using CUDA to run on the GPU. All data is allocated on the GPU but, due to optimzations performed by the CUDA compiler, virtual functions potentially caused issues. I avoided using virtual functions instead.
- [Sampling](_PROJECT/source/Raytracer/Sampling.h): importance sampling based on BRDF. Normalization happens in the actul tracing.
- [BRDF's](_PROJECT/source/Raytracer/BRDF.h): all BRDF functions used in this ray tracer.
- [Resource Buffer](_PROJECT/source/Raytracer/Buffers.h): Buffer class to easily transfer memory from GPU to CPU and vice versa. It uses **Unified Virtual Addressing** to automatically figure out the location (GPU or CPU) of the referenced data. It also make memory management easier.
- [Intersection Algorithmes](_PROJECT/source/Raytracer/Intersections.h): implementation of intersection algorithms for most used geometric entities
