#pragma once
#include "GPUHelpers.h"
#include "BRDF.h"

enum class MaterialType
{
	NONE,
	LAMBERT,
	LAMBERT_PHONG,
	COOK_TORRANCE
};

struct MaterialData
{
	MaterialType type = MaterialType::NONE;

	GPU_CALLABLE MaterialData(MaterialType _type) :
		type(_type)
	{}
};

struct LambertData final : public MaterialData
{
	RGBColor diffuseReflectance = {};

	GPU_CALLABLE LambertData(const RGBColor& _diffuseReflectance) :
		MaterialData(MaterialType::LAMBERT),
		diffuseReflectance(_diffuseReflectance)
	{}
};

struct LambertPhongData final : public MaterialData
{
	RGBColor diffuseReflectance = {};
	RGBColor specularReflectance = {};
	float specularExponent = 0.f;

	GPU_CALLABLE LambertPhongData(const RGBColor& _diffuseReflectance, const RGBColor& _specularReflectance,
		float _specularExponent) :
		MaterialData(MaterialType::LAMBERT_PHONG),
		diffuseReflectance(_diffuseReflectance), specularReflectance(_specularReflectance),
		specularExponent(_specularExponent)
	{}
};

struct CookTorranceData final : public MaterialData
{
	RGBColor reflectance = {};
	float metalness = 0.f;
	float roughness = 0.f;

	GPU_CALLABLE CookTorranceData(const RGBColor& _reflectance, float _metalness, float _roughness) :
		MaterialData(MaterialType::COOK_TORRANCE),
		reflectance(_reflectance), metalness(_metalness), roughness(_roughness)
	{}
};

GPU_CALLABLE static RGBColor ShadeMaterial(const MaterialData* data,
	const FVector3& hitNormal, const FVector3& lightDirection, const FVector3& viewDirection)
{
	//Local Variables
	RGBColor materialColor = {};
	const LambertData* lambertData = nullptr;
	const LambertPhongData* lambertPhongData = nullptr;
	const CookTorranceData* ctData = nullptr;

	//Cook Torrance Variables
	RGBColor f0 = RGBColor(0.04f, 0.04f, 0.04f); //Default dielectric surface reflectivity -> Plastic
	RGBColor F = {};
	RGBColor kd = {};
	RGBColor specular = {};
	RGBColor diffuse = {};
	FVector3 halfVector = {};
	float D = 0.f;
	float G = 0.f;
	float denom = 0.f;

	switch (data->type)
	{
	case MaterialType::LAMBERT:
		lambertData = (LambertData*)data;
		materialColor = Lambert(lambertData->diffuseReflectance);
		break;
	case MaterialType::LAMBERT_PHONG:
		lambertPhongData = (LambertPhongData*)data;
		materialColor = Lambert(lambertPhongData->diffuseReflectance)
			+ Phong(lambertPhongData->specularReflectance, lambertPhongData->specularExponent, 
				lightDirection, viewDirection, hitNormal);
		break;
	case MaterialType::COOK_TORRANCE:
		ctData = (CookTorranceData*)data;
		f0 = RGBColor(Lerp(f0.r, ctData->reflectance.r, ctData->metalness), //Pick either the default surface reflectivity of dielectric, or albedo of conductor, based on metalness parameter
			Lerp(f0.g, ctData->reflectance.g, ctData->metalness),
			Lerp(f0.b, ctData->reflectance.b, ctData->metalness));

		//---- Cook-Torrance Specular ---
		halfVector = viewDirection + lightDirection; //Normalized LightDirection + Normalized ViewDirection
		Normalize(halfVector);
		F = FresnelSchlick(halfVector, viewDirection, f0);
		D = NormalDistributionGGX(hitNormal, halfVector, ctData->roughness);
		G = GeometrySmith(hitNormal, viewDirection, lightDirection, ctData->roughness);

		denom = 4.f * std::max(Dot(hitNormal, viewDirection), 0.f) * std::max(Dot(hitNormal, lightDirection), 0.f);
		specular += (F * D * G) / std::max(denom, 0.0001f); //Prevent any divide by zero!

		//---- Lambert Diffuse ----
		kd = RGBColor(1.f, 1.f, 1.f) - F; //F == ks: specular reflection coefficient
		kd *= 1.f - ctData->metalness; //Nullify if the surface is metallic, because metallic surfaces don't refract light
		diffuse += Lambert(kd * ctData->reflectance);

		materialColor += diffuse + specular;
		break;
	default:
		break;
	}

	return materialColor;
}