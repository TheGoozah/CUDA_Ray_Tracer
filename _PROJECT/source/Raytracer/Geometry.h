#pragma once
#include "GPUHelpers.h"
#include "Math.h"
#include "Materials.h"

enum class GeometryType
{
	NONE,
	PLANE,
	SPHERE,
	TRIANGLE
};

struct ObjectData
{
	GeometryType type;
	MaterialData* materialData;

	GPU_CALLABLE ObjectData(GeometryType _type, MaterialData* _materialData) :
		type(_type), materialData(_materialData)
	{}
};

struct PlaneData final : public ObjectData
{
	FPoint3 position = {};
	FVector3 normal = {};

	GPU_CALLABLE PlaneData(const FPoint3& _position, const FVector3& _normal, MaterialData* _materialData) :
		ObjectData(GeometryType::PLANE, _materialData), position(_position), normal(_normal)
	{}
};

struct SphereData final : public ObjectData
{
	FPoint3 position = {};
	float radius = 0.f;

	GPU_CALLABLE SphereData(const FPoint3& _position, float _radius, MaterialData* _materialData) :
		ObjectData(GeometryType::SPHERE, _materialData), position(_position), radius(_radius)
	{}
};

struct TriangleData final : public ObjectData
{
	FPoint3 v0, v1, v2 = {};
	FVector3 normal = {};

	GPU_CALLABLE TriangleData(const FPoint3& _v0, const FPoint3& _v1, const FPoint3& _v2, MaterialData* _materialData) :
		ObjectData(GeometryType::TRIANGLE, _materialData), v0(_v0), v1(_v1), v2(_v2)
	{
		//Calculate normal + capture size of parallelogram created by the vertices (size of cross equals size of parallelogram)
		FVector3 edgeV0V1 = v0 - v1;
		FVector3 edgeV0V2 = v2 - v0;
		normal = Cross(edgeV0V1, edgeV0V2); //Not normalized!
	}
};