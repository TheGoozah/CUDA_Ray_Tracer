#pragma once
#include "Buffers.h"
#include "Window.h"
#include "Camera.h"
#include "SceneGraph.h"
#include "MaterialManager.h"

#pragma warning(push)
#pragma warning(disable: 26451)
#include "curand_kernel.h"
#pragma warning(pop)

struct RenderParameters
{
	uint32_t amountAccumulatedSamples = 0;
	uint32_t maxAmountAccumulatedSamples = 65536;
	uint32_t amountBouncesPerHit = 2;
	uint32_t amountIndirectSamplesPerHit = 1;
};

struct DeviceContext final
{
	//CPU Buffers
	std::unique_ptr<GlobalResourceBuffer<CameraData, BufferType::CPU>> cpuCameraData = nullptr;
	std::unique_ptr<GlobalResourceBuffer<RenderParameters, BufferType::CPU>> cpuRenderParameters = nullptr;
	//GPU Buffers
	std::unique_ptr<GlobalResourceBuffer<CameraData, BufferType::GPU>> gpuCameraData = nullptr;
	std::unique_ptr<GlobalResourceBuffer<SceneGraph, BufferType::GPU>> gpuSceneGraph = nullptr;
	std::unique_ptr<GlobalResourceBuffer<MaterialManager, BufferType::GPU>> gpuMaterialManager = nullptr;
	std::unique_ptr<GlobalResourceBuffer<curandState, BufferType::GPU>> gpuRandomNumberStates = nullptr;
	std::unique_ptr<GlobalResourceBuffer<RenderParameters, BufferType::GPU>> gpuRenderParameters = nullptr;

	DeviceContext(const WindowSettings& cpuWindowSettings);
	~DeviceContext();

	DeviceContext(DeviceContext&&) = default;
	DeviceContext& operator=(DeviceContext&&) = default;

	bool Update(const WindowSettings& cpuWindowSettings, Camera* cpuCamera, const RenderParameters& renderParams);

private:
	//Disable the possibility to allocate on the heap. This ensures it is not possible to create 
	//a smart pointer that might potentially call the destructor and clearing the GPU resources
	//by accident!
	void* operator new(size_t size) { return nullptr; };

	//Disable copy and assignment
	DeviceContext(DeviceContext&) = delete;
	DeviceContext& operator=(DeviceContext&) = delete;
};