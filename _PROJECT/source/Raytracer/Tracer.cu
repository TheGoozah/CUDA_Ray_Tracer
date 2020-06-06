#include "Tracer.cuh"
#include "Buffers.h"

#pragma warning(push)
#pragma warning(disable: 26812 26451)
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"
#pragma warning(pop)

//TEST MATH
#include "Math.h"
#include "RGBColor.h"
#include "Intersections.h"
#include "Sampling.h"

GPU_CALLABLE HitRecord TracePrimaryRay(const cc::vector<ObjectData*>& objects, const Ray& ray)
{
	//To prevent error/warnings, init possible variables
	SphereData* sphereData = nullptr;
	PlaneData* planeData = nullptr;
	TriangleData* triangleData = nullptr;

	HitRecord closestHitRecord = {};
	for (uint32_t i = 0; i < objects.size(); ++i)
	{
		ObjectData* objectData = objects.at(i);
		HitRecord hitRecord = {};

		switch (objectData->type)
		{
		case GeometryType::SPHERE:
			sphereData = (SphereData*)objectData;
			if (sphereData && RaySphereIntersection(ray, *sphereData, hitRecord)
				&& (hitRecord.t < closestHitRecord.t))
			{
				closestHitRecord = hitRecord;
			}
			break;
		case GeometryType::PLANE:
			planeData = (PlaneData*)objectData;
			if (planeData && PlaneRayIntersection(ray, *planeData, hitRecord)
				&& (hitRecord.t < closestHitRecord.t))
			{
				closestHitRecord = hitRecord;
			}
			break;
		case GeometryType::TRIANGLE:
			triangleData = (TriangleData*)objectData;
			if (triangleData && RayTriangleIntersection(ray, *triangleData, hitRecord)
				&& (hitRecord.t < closestHitRecord.t))
			{
				closestHitRecord = hitRecord;
			}
			break;
		case GeometryType::NONE:
			break;
		}
	}

	return closestHitRecord;
}

GPU_CALLABLE bool TraceShadowRay(const cc::vector<ObjectData*>& objects, const Ray& shadowRay)
{
	//To prevent error/warnings, init possible variables
	SphereData* sphereData = nullptr;
	PlaneData* planeData = nullptr;
	TriangleData* triangleData = nullptr;

	HitRecord shadowRecord = {};
	for (uint32_t i = 0; i < objects.size(); ++i)
	{
		ObjectData* objectData = objects.at(i);
		switch (objectData->type)
		{
		case GeometryType::SPHERE:
			sphereData = (SphereData*)objectData;
			if (sphereData && RaySphereIntersection(shadowRay, *sphereData, shadowRecord))
				return true;
			break;
		case GeometryType::PLANE:
			planeData = (PlaneData*)objectData;
			if (planeData && PlaneRayIntersection(shadowRay, *planeData, shadowRecord))
				return true;
		case GeometryType::TRIANGLE:
			triangleData = (TriangleData*)objectData;
			if (triangleData && RayTriangleIntersection(shadowRay, *triangleData, shadowRecord))
				return true;
		}
	}
	return false;
}

GPU_CALLABLE RGBColor RenderPixel(const cc::vector<ObjectData*>& objects, const cc::vector<LightData*>& lights,
	const Ray& ray, uint8_t depth, const RenderParameters* renderParams, curandState* randomState)
{
	//Exit after x amount of bounces, stop sampling (TODO: add russian roulette, change the accumulation buffer accordingly)
	if (depth >= renderParams->amountBouncesPerHit)
		return RGBColor(0.f, 0.f, 0.f);

	//Get closest hit (trace ray)
	HitRecord hitRecord = TracePrimaryRay(objects, ray);
	//if no intersection, stop because no bounces necessary
	if (!hitRecord.didHitObject)
		return RGBColor(0.f, 0.f, 0.f);

	//If intersection:
	//	- Do 1 indirect lighting pass on all lights for now
	//	- Do x amount of indirect samples. For every sample, trace that ray (= bounce)
	//--- DIRECT ---
	RGBColor brdf = {};
	RGBColor directLighting = {};
	for (uint32_t i = 0; i < lights.size(); ++i)
	{
		LightData* lightData = lights.at(i);
		FVector3 lightDirection = GetVectorToLight(lightData, hitRecord.point);

		//Check if direct light is visible
		float distancePointToLight = Normalize(lightDirection);
		float offset = 0.0001f;
		Ray shadowRay = Ray(hitRecord.point + (hitRecord.normal * offset),
			lightDirection, 0.f, distancePointToLight - offset);
		bool isVisible = !TraceShadowRay(objects, shadowRay);

		if (isVisible)
		{
			float lightArea = 1.f; //Make function per light type (needed for area lights obviously!)
			float solidAngleLight = lightArea / Square(distancePointToLight);
			directLighting +=
				lightData->color * lightData->intensity * solidAngleLight
				* Dot(hitRecord.normal, lightDirection);
		}

		brdf += ShadeMaterial(hitRecord.materialData, hitRecord.normal, lightDirection, -ray.direction);
	}
	directLighting /= lights.size();
	brdf /= lights.size();

	//--- INDIRECT ---
	RGBColor indirectDiffuse = {};
	RGBColor indirectSpecular = {};
	for (uint32_t amountSamples = 0; amountSamples < renderParams->amountIndirectSamplesPerHit; ++amountSamples)
	{
		float rand1 = curand_uniform(randomState);
		float rand2 = curand_uniform(randomState);
		if (hitRecord.materialData->type == MaterialType::LAMBERT)
		{
			//Diffuse
			FVector3 sampledDirection = SampleCosineWeightedHemisphere(rand1, rand2);
			FMatrix3 tangentHitPoint = ONBFromVector3(hitRecord.normal);
			sampledDirection = tangentHitPoint * sampledDirection;
			float pdf = PDFSampleCosineWeightedHemisphere(hitRecord.normal, sampledDirection);

			FPoint3 offsetPoint = hitRecord.point + (hitRecord.normal * 0.001f);
			Ray bounceRay = Ray(offsetPoint, sampledDirection);
			indirectDiffuse += RenderPixel(objects, lights, bounceRay, depth + 1, renderParams, randomState)
				/ pdf;
		}
		else if (hitRecord.materialData->type == MaterialType::LAMBERT_PHONG)
		{
			//Data
			LambertPhongData* lambertPhongData = (LambertPhongData*)hitRecord.materialData;
			RGBColor specularReflectanceCoefficient = lambertPhongData->specularReflectance;
			RGBColor diffuseReflectanceCoefficient = RGBColor(1.f, 1.f, 1.f) - specularReflectanceCoefficient;

			//Diffuse
			FVector3 sampledDirection = SampleCosineWeightedHemisphere(rand1, rand2);
			FMatrix3 tangentHitPoint = ONBFromVector3(hitRecord.normal);
			sampledDirection = tangentHitPoint * sampledDirection;
			float pdf = PDFSampleCosineWeightedHemisphere(hitRecord.normal, sampledDirection);

			FPoint3 offsetPoint = hitRecord.point + (hitRecord.normal * 0.001f);
			Ray bounceRay = Ray(offsetPoint, sampledDirection);
			indirectDiffuse += RenderPixel(objects, lights, bounceRay, depth + 1, renderParams, randomState)
				/ pdf;
			indirectDiffuse *= diffuseReflectanceCoefficient;

			//Specular Reflection
			sampledDirection = SampleHemisphereCosineLobe(rand1, rand2, lambertPhongData->specularExponent);
			tangentHitPoint = ONBFromVector3(hitRecord.normal);
			sampledDirection = tangentHitPoint * sampledDirection;

			FVector3 incomingDirection = -ray.direction;
			FVector3 reflectedSample = Reflect(incomingDirection + sampledDirection, hitRecord.normal);
			Normalize(reflectedSample);
			FVector3 perfectReflection = Reflect(incomingDirection, hitRecord.normal);
			Normalize(perfectReflection);

			pdf = PDFSampleHemisphereCosineLobe(perfectReflection, reflectedSample, lambertPhongData->specularExponent);
			float specNorm = (lambertPhongData->specularExponent + 2) / (lambertPhongData->specularExponent + 1);
			float cosA = std::powf(Dot(perfectReflection, reflectedSample), lambertPhongData->specularExponent);

			offsetPoint = hitRecord.point + (hitRecord.normal * 0.001f);
			bounceRay = Ray(offsetPoint, reflectedSample);
			RGBColor spec = RenderPixel(objects, lights, bounceRay, depth + 1, renderParams, randomState) * cosA
				* specNorm / pdf;
			indirectSpecular = spec * specularReflectanceCoefficient * lambertPhongData->specularReflectance;
		}
		else if (hitRecord.materialData->type == MaterialType::COOK_TORRANCE)
		{
			//Data
			CookTorranceData* ctData = (CookTorranceData*)hitRecord.materialData;
			RGBColor f0 = RGBColor(0.04f, 0.04f, 0.04f); //Default dielectric surface reflectivity -> Plastic
			f0 = RGBColor(Lerp(f0.r, ctData->reflectance.r, ctData->metalness), //Pick either the default surface reflectivity of dielectric, or albedo of conductor, based on metalness parameter
				Lerp(f0.g, ctData->reflectance.g, ctData->metalness),
				Lerp(f0.b, ctData->reflectance.b, ctData->metalness));
			const RGBColor F = FresnelSchlickRoughness(hitRecord.normal, -ray.direction, f0, ctData->roughness);

			//Diffuse
			FVector3 sampledDirection = SampleCosineWeightedHemisphere(rand1, rand2);
			FMatrix3 tangentHitPoint = ONBFromVector3(hitRecord.normal);
			sampledDirection = tangentHitPoint * sampledDirection;
			float pdf = PDFSampleCosineWeightedHemisphere(hitRecord.normal, sampledDirection);

			FPoint3 offsetPoint = hitRecord.point + (hitRecord.normal * 0.001f);
			Ray bounceRay = Ray(offsetPoint, sampledDirection);
			RGBColor diff = RenderPixel(objects, lights, bounceRay, depth + 1, renderParams, randomState)
				/ pdf;
			RGBColor kd = (RGBColor(1.f, 1.f, 1.f) - F) * (1.f - ctData->metalness);
			indirectDiffuse += kd * diff * ctData->reflectance;

			//Specular Reflection
			sampledDirection = SampleCookTorrance(rand1, rand2, ctData->roughness);
			tangentHitPoint = ONBFromVector3(hitRecord.normal);
			sampledDirection = tangentHitPoint * sampledDirection;

			FVector3 incomingDirection = -ray.direction;
			FVector3 reflectedSample = Reflect(incomingDirection + sampledDirection, hitRecord.normal);
			Normalize(reflectedSample);

			pdf = PDFCookTorrance(hitRecord.normal, -ray.direction, reflectedSample, ctData->roughness);
			float cosTheta2 = Dot(hitRecord.normal, reflectedSample);
			float a = ctData->roughness; //TODO:check alpha = Square(Square(ctData->roughness));
			float exp = a / cosTheta2;
			float rt = (1 + exp) * cosTheta2;
			float norm = 1.f / (PI * ctData->roughness * Square(rt));

			offsetPoint = hitRecord.point + (hitRecord.normal * 0.001f);
			bounceRay = Ray(offsetPoint, reflectedSample);
			RGBColor spec = RenderPixel(objects, lights, bounceRay, depth + 1, renderParams, randomState) *
				norm / pdf;
			indirectSpecular += spec * f0;
		}
	}
	indirectDiffuse /= static_cast<float>(renderParams->amountIndirectSamplesPerHit);
	indirectSpecular /= static_cast<float>(renderParams->amountIndirectSamplesPerHit);

	//+ Calculate final light contribution at this point
	return (directLighting + indirectDiffuse) * brdf + indirectSpecular;
}

__global__ void RenderGPU(RGBColor* accumulationBuffer, uint32_t* pixels, uint32_t width, uint32_t height,
	const RenderParameters* renderParameters,
	const CameraData* cameraData, SceneGraph* sceneGraph, curandState* randomStates)
{
	//https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
	int xID = threadIdx.x + blockIdx.x * blockDim.x;
	int yID = threadIdx.y + blockIdx.y * blockDim.y;
	int gID = xID + yID * blockDim.x * gridDim.x;

	curandState localRandomState = randomStates[gID];

	const cc::vector<ObjectData*>& objects = sceneGraph->GetObjects();
	const cc::vector<LightData*>& lights = sceneGraph->GetLights();
	const uint32_t windowSize = height * width;

	if (gID < windowSize)
	{
		RGBColor finalColor = {};

		float x = (2.f * ((xID + 0.5f) / width) - 1.f) * cameraData->fov * cameraData->aspectRatio;
		float y = (1.f - 2.f * ((yID + 0.5f) / height)) * cameraData->fov;
		FPoint4 pixelPositionWorld = cameraData->ONB * FPoint4(x, y, -1.f);
		FVector3 rayDirection = FVector3(pixelPositionWorld - FPoint4(cameraData->position));
		Normalize(rayDirection);
		Ray ray = Ray(cameraData->position, rayDirection);

		uint8_t depth = 0;
		finalColor = RenderPixel(objects, lights, ray, depth, renderParameters, &localRandomState);

		//Save state for random sampling
		randomStates[gID] = localRandomState;

		//Fill accumulation buffer, fill pixel buffer + save amount samples
		RGBColor accColor = accumulationBuffer[gID];
		accColor += finalColor;
		accumulationBuffer[gID] = accColor;

		accColor /= (float)renderParameters->amountAccumulatedSamples;
		accColor = GammaCorrectionSRGB(accColor);
		pixels[gID] = GetSDL_ARGBColor(accColor);
	}
}

void Render(RGBColor* accumulationBuffer, uint32_t* pixels, uint32_t width, uint32_t height, DeviceContext& deviceContext)
{
	//TEST SCHEDULING
	dim3 threadsPerBlock(16, 16); //256 threads in total	   
	dim3 amountBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y); //Would not work if not power of 2!

	RenderGPU << <amountBlocks, threadsPerBlock >> > (accumulationBuffer, pixels, width, height,
		deviceContext.gpuRenderParameters->GetRawBuffer(),
		deviceContext.gpuCameraData->GetRawBuffer(),
		deviceContext.gpuSceneGraph->GetRawBuffer(),
		deviceContext.gpuRandomNumberStates->GetRawBuffer());
	cudaDeviceSynchronize();
}