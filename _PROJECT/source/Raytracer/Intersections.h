#pragma once
#include "Math.h"
#include "Geometry.h"
#include "Materials.h"

struct Ray final
{
	//=== Datamembers ===
	FPoint3 origin = {};
	FVector3 direction = {};
	float tMin = 0.0001f;
	float tMax = FLT_MAX;

	//=== Constructors & Destructor ===
	BOTH_CALLABLE Ray() {}
	BOTH_CALLABLE Ray(const FPoint3& _origin, const FVector3& _direction, float _tMin = 0.0001f, float _tMax = FLT_MAX)
		:origin(_origin), direction(_direction), tMin(_tMin), tMax(_tMax) {}
	BOTH_CALLABLE Ray(const Ray& ray)
		: origin(ray.origin), direction(ray.direction), tMin(ray.tMin), tMax(ray.tMax) {}
	BOTH_CALLABLE Ray(Ray&& ray) noexcept
		:origin(std::move(ray.origin)), direction(std::move(ray.direction)), tMin(std::move(ray.tMin)), tMax(std::move(ray.tMax)) {}
	BOTH_CALLABLE ~Ray() {};

	//=== Operators ===
	BOTH_CALLABLE Ray& operator=(const Ray& ray)
	{ origin = ray.origin; direction = ray.direction; tMin = ray.tMin; tMax = ray.tMax; return *this; }
	BOTH_CALLABLE Ray& operator=(Ray&& ray) noexcept
	{ origin = std::move(ray.origin); direction = std::move(ray.direction); tMin = std::move(ray.tMin); tMax = std::move(ray.tMax); return *this; }
};

struct HitRecord final
{
	FPoint3 point = {}; //point where the object was hit (origin + t * direction)
	FVector3 normal = {}; //normal at the point
	float t = FLT_MAX; //distance along the ray the object was hit
	bool didHitObject = false; //bool to store if we actually hit an object
	MaterialData* materialData = {}; //pointer to data material
};

GPU_CALLABLE bool PlaneRayIntersection(const Ray& ray, const PlaneData& planeData, HitRecord& hitRecord)
{
	FVector3 L = planeData.position - ray.origin;
	double t = Dot(L, planeData.normal) / Dot(ray.direction, planeData.normal);
	if (t >= ray.tMin && t <= ray.tMax)
	{
		hitRecord.t = static_cast<float>(t);
		hitRecord.point = ray.origin + static_cast<float>(t)* ray.direction;
		hitRecord.normal = planeData.normal;
		hitRecord.didHitObject = true;
		hitRecord.materialData = planeData.materialData;
		return true;
	}
	return false;
}

GPU_CALLABLE bool RaySphereIntersection(const Ray& ray, const SphereData& sphereData, HitRecord& hitRecord)
{
	FVector3 originToCenter = sphereData.position - ray.origin;
	double tca = Dot(originToCenter, ray.direction);
	double od2 = SqrMagnitude(Reject(originToCenter, ray.direction));
	float s2 = Square(sphereData.radius);
	if (od2 > s2)
		return false;

	double thc = sqrt(s2 - od2);
	double t0 = tca - thc; //Distance origin ray to first hit
	double t1 = tca + thc; //Distance origin ray to second hit

	if (t0 > t1) //Get closest hit
		Swap(t0, t1);
	if (t0 < 0) //Check if there is an actual hit
	{
		t0 = t1;
		if (t0 < 0)
			return false;
	}

	if (t0 < ray.tMin || t0 > ray.tMax) //Check within ray range
		return false;

	hitRecord.t = static_cast<float>(t0);
	hitRecord.point = ray.origin + static_cast<float>(t0)* ray.direction;
	hitRecord.normal = GetNormalized(hitRecord.point - sphereData.position);
	hitRecord.materialData = sphereData.materialData;
	hitRecord.didHitObject = true;
	return true;
}

GPU_CALLABLE bool RayTriangleIntersection(const Ray& ray, const TriangleData& triangleData, HitRecord& hitRecord)
{
	//Set hitRecord to false
	hitRecord.didHitObject = false;

	//Edge case 1: when ray and triangle are parallel, there is no hit!
	float nDotr = Dot(triangleData.normal, ray.direction);
	if (AreEqual(nDotr, 0.f))
		return false;

	//Culling
	if (nDotr < 0.f) //Backface culling
		return false;

	//Calculate t
	FPoint3 center = FPoint3((FVector3(triangleData.v0) + FVector3(triangleData.v1) + FVector3(triangleData.v2)) / 3.f);
	FVector3 L = center - ray.origin;
	float t = Dot(L, triangleData.normal) / Dot(ray.direction, triangleData.normal);

	//Within range
	if (t < ray.tMin || t > ray.tMax)
		return false;

	//Calculate intersection point on parallelogram
	FPoint3 p = ray.origin + t * ray.direction;

	//Check if this point is inside the triangle! Called the Inside-Outside test.
	FVector3 c = {};
	FVector3 side = {};
	FVector3 pointToSide = {};

	//Edge A
	side = triangleData.v1 - triangleData.v0;
	pointToSide = p - triangleData.v0;
	c = Cross(side, pointToSide);
	if (Dot(triangleData.normal, c) > 0.f) //point is on the right side, so outside the triangle
		return false;
	//Edge B
	side = triangleData.v2 - triangleData.v1;
	pointToSide = p - triangleData.v1;
	c = Cross(side, pointToSide);
	if (Dot(triangleData.normal, c) > 0.f) //point is on the right side, so outside the triangle
		return false;
	//Edge C
	side = triangleData.v0 - triangleData.v2;
	pointToSide = p - triangleData.v2;
	c = Cross(side, pointToSide);
	if (Dot(triangleData.normal, c) > 0.f) //point is on the right side, so outside the triangle
		return false;

	//If the inside-outside test succeeded, we have hit in our triangle, so store info
	hitRecord.t = static_cast<float>(t);
	hitRecord.point = p;
	hitRecord.normal = -GetNormalized(triangleData.normal);
	hitRecord.materialData = triangleData.materialData;
	hitRecord.didHitObject = true;
	return true;
}