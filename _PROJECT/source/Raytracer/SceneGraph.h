#pragma once
#include "Containers.h"
#include "Geometry.h"
#include "Lights.h"

class SceneGraph final
{
	cc::vector<ObjectData*> m_Objects = cc::vector<ObjectData*>(10);
	cc::vector<LightData*> m_Lights = cc::vector<LightData*>(10);

public:
	GPU_CALLABLE SceneGraph() 
	{};
	GPU_CALLABLE ~SceneGraph() 
	{};

	GPU_CALLABLE void Clear()
	{
		m_Objects.destroy();
		m_Lights.destroy();
	}

	GPU_CALLABLE void AddObjectToScene(ObjectData* obj)
	{
		//Taking over ownership! Don't forget to clear the graph, which will destroy the container.
		//TODO: implement own shared_ptr with __device__ support
		m_Objects.push_back(obj);
	}

	GPU_CALLABLE void AddLightToScene(LightData* light)
	{
		//Taking over ownership! Don't forget to clear the graph, which will destroy the container.
		//TODO: implement own shared_ptr with __device__ support
		m_Lights.push_back(light);
	}

	GPU_CALLABLE const cc::vector<ObjectData*>& GetObjects() const
	{ return m_Objects; }

	GPU_CALLABLE const cc::vector<LightData*>& GetLights() const
	{ return m_Lights; }
};