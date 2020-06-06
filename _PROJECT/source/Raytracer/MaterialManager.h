#pragma once
#include "Containers.h"
#include "Materials.h"

class MaterialManager final
{
	cc::vector<MaterialData*> m_Materials = cc::vector<MaterialData*>(10);

public:
	GPU_CALLABLE MaterialManager()
	{};
	GPU_CALLABLE ~MaterialManager()
	{};

	GPU_CALLABLE void Clear()
	{
		m_Materials.destroy();
	}

	GPU_CALLABLE MaterialData* AddMaterial(MaterialData* mat)
	{
		//Taking over ownership! Don't forget to clear the graph, which will destroy the container.
		//TODO: implement own shared_ptr with __device__ support
		m_Materials.push_back(mat);
		return mat;
	}
};