#pragma once
#include "GPUHelpers.h"
#include "Math.h"
#include "BRDF.h"
#include <algorithm>

//See page 19-20 of https://www.ii.uni.wroc.pl/~anl/cgfiles/TotalCompendium.pdf
//Uniform distributed random values ur0 and ur1, range [0,1],
//used to sample uniformly from a disk and then project the sample
//onto the hemisphere. Don't forgot to put this point in the correct tangent space!
GPU_CALLABLE FVector3 SampleHemisphere(float ur0, float ur1)
{
	const float r = sqrtf(1.f - std::pow(ur1, 2.f));
	const float phi = 2.f * PI * ur0;

	const float x = cos(phi) * r;
	const float y = sin(phi) * r;
	const float z = ur1;
	return FVector3(x, y, z);
}

GPU_CALLABLE float PDFSampleHemisphere()
{
	return 1.f / (2.f * PI);
}

GPU_CALLABLE FVector3 SampleCosineWeightedHemisphere(float ur0, float ur1)
{
	const float r = sqrtf(ur1);
	const float phi = 2.f * PI * ur0;
	const float x = cos(phi) * r;
	const float y = sin(phi) * r;
	const float z = sqrtf(1.f - ur1);
	return FVector3(x, y, z);
}

GPU_CALLABLE float PDFSampleCosineWeightedHemisphere(const FVector3& normal, const FVector3& sampledDirection)
{
	return std::max(Dot(normal, sampledDirection) / (float)PI, .001f);
}

GPU_CALLABLE FVector3 SampleHemisphereCosineLobe(float ur0, float ur1, float exponent)
{
	const float phi = 2.f * (float)PI * ur0;
	const float r = sqrtf(1.f - std::pow(ur1, 2.f / (exponent + 1)));
	const float x = cos(phi) * r;
	const float y = sin(phi) * r;
	const float z = std::pow(ur1, 1.f / (exponent + 1));
	return FVector3(x, y, z);
}

GPU_CALLABLE float PDFSampleHemisphereCosineLobe(const FVector3& perfectReflectionDirection, 
	const FVector3& sampledReflectionDirection, float exponent)
{
	const float cosTheta = Dot(sampledReflectionDirection, perfectReflectionDirection);
	const float pdf = (exponent + 1.f) / (2.f * (float)PI) * std::powf(cosTheta, exponent);
	return std::max(pdf, .001f);
}

GPU_CALLABLE FVector3 SampleCookTorrance(float ur0, float ur1, float roughness)
{
	const float a = std::max(Square(Square(roughness)), .001f);
	const float phi = 2.f * (float)PI * ur0;
	const float cosTheta = sqrtf((1.f - ur1) / (1.f + (Square(a) - 1.f) * ur1));
	const float sinTheta = Clamp(sqrtf(1.f - Square(cosTheta)), 0.f, 1.f);
	const float sinPhi = sin(phi);
	const float cosPhi = cos(phi);
	const float x = cos(phi) * sinTheta;
	const float y = sin(phi) * sinTheta;
	const float z = cosTheta;
	return FVector3(x, y, z);
}

GPU_CALLABLE float PDFCookTorrance(const FVector3& normal, const FVector3& incomingDirection, const FVector3 sampledReflectionDirection, float roughness)
{
	const FVector3 halfVector = GetNormalized(sampledReflectionDirection + incomingDirection);
	const float NdotH = Dot(normal, halfVector);
	const float D = NormalDistributionGGX(normal, halfVector, roughness);
	return (D * NdotH) / (4.f * std::abs(Dot(sampledReflectionDirection, halfVector)));
}

//Create ONB, usually from normal vector.
GPU_CALLABLE FMatrix3 ONBFromVector3(const FVector3& n)
{
	//Check for up vector to get best direction, slide 15
	//http://www.cs.uu.nl/docs/vakken/magr/2017-2018/slides/lecture%2013%20-%20BRDFs.pdf
	const FVector3 up = std::abs(n.x) > 0.99f ? FVector3(0.f, 1.f, 0.f) : FVector3(1.f, 0.f, 0.f);
	const FVector3 tangentZ = GetNormalized(n);
	const FVector3 tangentX = GetNormalized(Cross(up, tangentZ));
	const FVector3 tangentY = GetNormalized(Cross(tangentZ, tangentX));
	return FMatrix3(tangentX, tangentY, tangentZ);
}