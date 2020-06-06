#pragma once
#include <stdint.h>
#include "DeviceContext.cuh"

__global__ void RenderPixel(RGBColor* accumulationBuffer, uint32_t* pixels, uint32_t width, uint32_t height, 
	const RenderParameters* renderParameters,
	const CameraData* cameraData, SceneGraph* sceneGraph, curandState* randomStates);

void Render(RGBColor* accumulationBuffer, uint32_t* pixels, uint32_t width, uint32_t height, DeviceContext& deviceContext);