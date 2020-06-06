#pragma once
#include "GPUHelpers.h"
#include "RGBColor.h"
#include "Math.h"

//kd: diffuse reflectance coefficient
GPU_CALLABLE static RGBColor Lambert(const RGBColor& kd)
{
	return kd / static_cast<float>(PI);
}

//ks: specular reflectance coefficient
//exp: Phong exponent
//l: incoming (incident) light direction
//v: view direction
//n: normal of surface
GPU_CALLABLE static RGBColor Phong(const RGBColor& ks, float exp, const FVector3& l, const FVector3& v, const FVector3& n)
{
	const FVector3 r = Reflect(l, n);
	const float rDv = Dot(r, v);
	RGBColor d = {};
	if (rDv > 0.f)
		d = ks * static_cast<float>(pow(rDv, exp));
	return d;
}

//h: normalized halfvector between view and light directions
//v: normalized view direction
//f0: base reflectivity of a surface based on IOR (indices of refraction) -> Is different for Dielectrics (Non-Metal) and Conductors (Metal)
GPU_CALLABLE static RGBColor FresnelSchlick(const FVector3& h, const FVector3& v, const RGBColor& f0)
{
	const float HdotV = std::max(Dot(h, v), 0.f);
	return f0 + (RGBColor(1.f, 1.f, 1.f) - f0) * powf(1.f - HdotV, 5.f);
}

//n: normal surface
//v: normalized view direction
//f0: base reflectivity of a surface based on IOR
//roughness: roughness of the material
GPU_CALLABLE static RGBColor FresnelSchlickRoughness(const FVector3& n, const FVector3& v, const RGBColor& f0, float roughness)
{
	const float NdotV = std::max(Dot(n, v), 0.f);
	const RGBColor r = Max(RGBColor(1.f, 1.f, 1.f) - RGBColor(roughness, roughness, roughness), f0);
	return f0 + (r - f0) * powf(1.f - NdotV, 5.f);
}

//n: normal surface
//h: normalized halfvector between view and light directions
//roughness: roughness of the material -> we square this value to get alpa, conform UE4 implementation (not Disney)
GPU_CALLABLE static float NormalDistributionGGX(const FVector3& n, const FVector3& h, float roughness)
{
	const float a = Square(roughness);
	const float a2 = Square(a);
	const float NdotH = std::max(Dot(n, h), 0.f);
	return a2 / (static_cast<float>(PI)* Square((Square(NdotH) * (a2 - 1.f) + 1.f)));
}

//n: normal surface
//v: normalized view direction
//roughness: roughness of the material -> we square this value to get alpa, conform UE4 implementation (not Disney)
GPU_CALLABLE static float GeometrySchlickGGX(const FVector3& n, const FVector3& v, float roughness, bool directLighting = true)
{
	//Remap k
	float k = 0.f;
	if(directLighting)
		k = (Square(roughness + 1)) / 8.f; //Direct Lighting
	else
		k = Square(roughness) / 2.f; //IBL Lighting
	const float NdotV = std::max(Dot(n, v), 0.f);
	return NdotV / (NdotV * (1.f - k) + k);
}

//n: normal surface
//v: normalized view direction
//l: normalized light direction
//roughness: roughness of the material 
GPU_CALLABLE static float GeometrySmith(const FVector3& n, const FVector3& v, const FVector3& l, float roughness)
{
	const float ggx1 = GeometrySchlickGGX(n, v, roughness);
	const float ggx2 = GeometrySchlickGGX(n, l, roughness);
	return ggx1 * ggx2;
}