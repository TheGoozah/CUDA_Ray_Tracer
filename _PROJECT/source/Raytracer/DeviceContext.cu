#include "DeviceContext.cuh"
#include "Geometry.h"
#include "Lights.h"
#include "Materials.h"

__global__ void CreateDeviceContext(CameraData* cameraData, SceneGraph* sceneGraph, MaterialManager* materialManager)
{
	//Invoke the constructors on the pre-allocated memory using placement new operator
	//---- CAMERA ----
	cameraData = new (cameraData) CameraData();

	//---- MATERIAL MANAGER + MATERIALS ----
	materialManager = new (materialManager) MaterialManager();
	MaterialData* whiteLambert = materialManager->AddMaterial(new LambertData({ 1.f, 1.f, 1.f }));
	MaterialData* greyLambert = materialManager->AddMaterial(new LambertData({ 0.18f, 0.18f, 0.18f }));
	MaterialData* greenLambert = materialManager->AddMaterial(new LambertData({ 0.f, 1.f, 0.f }));
	MaterialData* redLambert = materialManager->AddMaterial(new LambertData({ 1.f, 0.f, 0.f }));
	MaterialData* greyLambertPhong = materialManager->AddMaterial(new LambertPhongData({ 0.25f, 0.25f, 0.25f },
		{ 0.75f, 0.75f, 0.75f }, 10.f));
	MaterialData* ctPlasticSmooth = materialManager->AddMaterial(new CookTorranceData({ 1.f, 0.f, 0.f }, 0.f, 0.4f));
	MaterialData* ctPlasticRough = materialManager->AddMaterial(new CookTorranceData({ 1.f, 1.f, 1.f }, 0.f, 0.8f));
	MaterialData* ctConductor = materialManager->AddMaterial(new CookTorranceData({ 0.98f, 0.82f, 0.75f }, 1.f, 0.8f));

	//---- SCENE ----
	sceneGraph = new (sceneGraph) SceneGraph();
	sceneGraph->AddObjectToScene(new SphereData(FPoint3(-1.75f, .75f, 0.f), .75f, ctConductor));
	sceneGraph->AddObjectToScene(new SphereData(FPoint3(0.f, .75f, 0.f), .75f, ctPlasticSmooth));
	sceneGraph->AddObjectToScene(new SphereData(FPoint3(2.5f, .75f, 0.f), .75f, ctPlasticRough));

	//Floor
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(-4.f, 0.f, -4.f), FPoint3(-4.f, 0.f, 4.f), FPoint3(4.f, 0.f, 4.f),
		greyLambert));
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(4.f, 0.f, 4.f), FPoint3(4.f, 0.f, -4.f), FPoint3(-4.f, 0.f, -4.f),
		greyLambert));
	//Left Wall
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(-4.f, 4.f, 4.f), FPoint3(-4.f, 0.f, 4.f), FPoint3(-4.f, 0.f, -4.f),
		redLambert));
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(-4.f, 0.f, -4.f), FPoint3(-4.f, 4.f, -4.f), FPoint3(-4.f, 4.f, 4.f),
		redLambert));
	//Right Wall
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(4.f, 4.f, 4.f), FPoint3(4.f, 0.f, -4.f), FPoint3(4.f, 0.f, 4.f), 
		greenLambert));
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(4.f, 0.f, -4.f), FPoint3(4.f, 4.f, 4.f), FPoint3(4.f, 4.f, -4.f),
		greenLambert));
	//Back Wall
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(-4.f, 4.f, -4.f), FPoint3(-4.f, 0.f, -4.f), FPoint3(4.f, 0.f, -4.f),
		greyLambert));
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(4.f, 0.f, -4.f), FPoint3(4.f, 4.f, -4.f), FPoint3(-4.f, 4.f, -4.f),
		greyLambert));
	//Roof
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(-4.f, 4.f, -4.f), FPoint3(4.f, 4.f, 4.f), FPoint3(-4.f, 4.f, 4.f),
		greyLambert));
	sceneGraph->AddObjectToScene(new TriangleData(FPoint3(4.f, 4.f, 4.f), FPoint3(-4.f, 4.f, -4.f), FPoint3(4.f, 4.f, -4.f),
		greyLambert));
	
	//sceneGraph->AddLightToScene(new PointLightData(FPoint3(-2.f, 2.5f, 2.f), RGBColor(1.f, 1.f, 1.f), 10.f));
	sceneGraph->AddLightToScene(new PointLightData(FPoint3(0.f, 2.5f, 2.f), RGBColor(1.f, 1.f, 1.f), 15.f));
}

__global__ void InitRandomStatesPerThread(curandState* state)
{
	int xID = threadIdx.x + blockIdx.x * blockDim.x;
	int yID = threadIdx.y + blockIdx.y * blockDim.y;
	int gID = xID + yID * blockDim.x * gridDim.x;
	curand_init(0, gID, 0, &state[gID]);
}

__global__ void ClearDeviceContext(SceneGraph* sceneGraph, MaterialManager* materialManager)
{
	//Clear content, will free device memory later through the actual buffer
	sceneGraph->Clear();
	materialManager->Clear();
}

DeviceContext::DeviceContext(const WindowSettings& cpuWindowSettings)
{
	//Create Buffers on CPU memory
	cpuCameraData = std::make_unique<GlobalResourceBuffer<CameraData, BufferType::CPU>>(1);
	cpuRenderParameters = std::make_unique<GlobalResourceBuffer<RenderParameters, BufferType::CPU>>(1);
	//Create buffers on GPU memory
	gpuCameraData = std::make_unique<GlobalResourceBuffer<CameraData, BufferType::GPU>>(1);
	gpuSceneGraph = std::make_unique<GlobalResourceBuffer<SceneGraph, BufferType::GPU>>(1);
	gpuMaterialManager = std::make_unique<GlobalResourceBuffer<MaterialManager, BufferType::GPU>>(1);
	gpuRandomNumberStates = std::make_unique<GlobalResourceBuffer<curandState, BufferType::GPU>>(
		cpuWindowSettings.width * cpuWindowSettings.height);
	gpuRenderParameters = std::make_unique<GlobalResourceBuffer<RenderParameters, BufferType::GPU>>(1);

	//Fill Buffers on GPU (forcing all internal allocations to be in device memory)
	CreateDeviceContext << <1, 1 >> > (
		gpuCameraData->GetRawBuffer(), 
		gpuSceneGraph->GetRawBuffer(),
		gpuMaterialManager->GetRawBuffer());
	cudaDeviceSynchronize();
	
	//Fill the random number states for every pixel
	//TODO: make sure the ShadePixel uses the same execution variables!!!
	dim3 threadsPerBlock(16, 16);
	dim3 amountBlocks(cpuWindowSettings.width / threadsPerBlock.x, cpuWindowSettings.height / threadsPerBlock.y);
	InitRandomStatesPerThread << <amountBlocks, threadsPerBlock >> > (gpuRandomNumberStates->GetRawBuffer());
	cudaDeviceSynchronize();
}

DeviceContext::~DeviceContext()
{
	ClearDeviceContext << <1, 1 >> > (gpuSceneGraph->GetRawBuffer(), gpuMaterialManager->GetRawBuffer());
	cudaDeviceSynchronize();
}

bool DeviceContext::Update(const WindowSettings& cpuWindowSettings, Camera* cpuCamera, const RenderParameters& renderParams)
{
	//Camera - reset in case values change!
	cpuCameraData.get()->GetRawBuffer()->fov = tan(ToRadians(cpuCamera->FOVAngle) / 2.f);
	cpuCameraData.get()->GetRawBuffer()->aspectRatio = (float)cpuWindowSettings.width / cpuWindowSettings.height;
	cpuCameraData.get()->GetRawBuffer()->position = cpuCamera->position;
	cpuCameraData.get()->GetRawBuffer()->ONB = cpuCamera->LookAt();
	CopyBuffers(gpuCameraData.get(), cpuCameraData.get());
	
	//Render Params - find better way to init
	cpuRenderParameters.get()->GetRawBuffer()->amountIndirectSamplesPerHit = renderParams.amountIndirectSamplesPerHit;

	//Reset sample if camera moved or changed
	if (cpuCamera->didChange ||
		(cpuRenderParameters.get()->GetRawBuffer()->amountBouncesPerHit != renderParams.amountBouncesPerHit))
	{
		cpuRenderParameters.get()->GetRawBuffer()->amountBouncesPerHit = renderParams.amountBouncesPerHit;
		cpuRenderParameters.get()->GetRawBuffer()->amountAccumulatedSamples = 0;
	}

	//Overflow of accumulation buffer safety
	if (cpuRenderParameters.get()->GetRawBuffer()->amountAccumulatedSamples >= renderParams.maxAmountAccumulatedSamples)
		return false;

	//Update sample info - based on camera changes
	cpuRenderParameters.get()->GetRawBuffer()->amountAccumulatedSamples += 1;
	CopyBuffers(gpuRenderParameters.get(), cpuRenderParameters.get());

	return true;
}
