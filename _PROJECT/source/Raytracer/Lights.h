#pragma once
#include "GPUHelpers.h"
#include "Math.h"
#include "RGBColor.h"

enum class LightType
{
	NONE,
	POINT,
	DIRECTIONAL
};

struct LightData
{
	RGBColor color = {};
	float intensity = 0.f;
	LightType type = LightType::NONE;

	GPU_CALLABLE LightData(const RGBColor& _color, const float _intensity, const LightType _lightType)
		:color(_color), intensity(_intensity), type(_lightType)
	{}
};

struct PointLightData final : public LightData
{
	FPoint3 position = {};
	GPU_CALLABLE PointLightData(const FPoint3& _position, const RGBColor& _color, const float _intensity)
		:LightData(_color, _intensity, LightType::POINT), 
		position(_position)
	{}
};

struct DirectionalLightData final : public LightData
{
	FVector3 direction = {};

	GPU_CALLABLE DirectionalLightData(const FVector3& _direction, const RGBColor& _color, const float _intensity) :
		LightData(_color, _intensity, LightType::DIRECTIONAL),
		direction(_direction)
	{}
};

GPU_CALLABLE static FVector3 GetVectorToLight(const LightData* data, const FPoint3& p)
{
	//To prevent error/warnings, init possible variables
	DirectionalLightData* directData = nullptr;
	PointLightData* pointData = nullptr;

	switch (data->type)
	{
	case LightType::DIRECTIONAL:
		directData = (DirectionalLightData*)data;
		return directData->direction;
		break;
	case LightType::POINT:
		pointData = (PointLightData*)data;
		return (pointData->position - p);
		break;
	case LightType::NONE:
		return FVector3::ZeroVector();
		break;
	}
	return FVector3::ZeroVector();
}