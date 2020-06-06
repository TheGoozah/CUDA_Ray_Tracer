#pragma once
#include <stdint.h>
#include <type_traits>
#include "GPUHelpers.h"

enum class BufferType
{
	GPU,
	CPU
};

template<typename T, BufferType type>
class GlobalResourceBuffer final
{
	BufferType m_Type;
	T* m_ResourceBuffer = nullptr;
	uint32_t m_AmountElements = 0;
	bool m_NeedToDelete = true;

public:
	GlobalResourceBuffer<T, type>(uint32_t amountElements):
		m_AmountElements(amountElements), m_Type(type), m_NeedToDelete(true)
	{
		uint32_t byteSize = GetTotalByteSize();
		switch (type)
		{
		case BufferType::GPU:
			GPUErrorCheck(cudaMalloc((T**)&m_ResourceBuffer, byteSize));
			GPUErrorCheck(cudaMemset(m_ResourceBuffer, 0, byteSize));
			break;
		case BufferType::CPU:
			m_ResourceBuffer = (T*)malloc(byteSize);
			memset(m_ResourceBuffer, 0, byteSize);
			break;
		}
	}

	//For pre-allocated resources (generally memory allocated by API's)
	GlobalResourceBuffer<T, type>(T* resource, uint32_t amountElements):
		m_Type(type), m_ResourceBuffer(resource), m_AmountElements(amountElements), m_NeedToDelete(false)
	{};

	~GlobalResourceBuffer<T, type>()
	{
		if (!m_NeedToDelete)
			return;

		switch (type)
		{
		case BufferType::GPU:
			GPUErrorCheck(cudaFree(m_ResourceBuffer));
			break;
		case BufferType::CPU:
			free(m_ResourceBuffer);
			break;
		}
	}

	template<uint32_t N, typename U>
	bool FillBuffer(U const (&data)[N])
	{
		if (!std::is_same<T, U>::value || N != m_AmountElements)
		{
			printf("ERROR: could not FillBuffer because of mismatching types and/or mismatch amount of elements");
			return false;
		}
		
		//Requires Unified Virtual Addressing support : cudaMemcpyDefault
		uint32_t byteSize = GetTotalByteSize();
		cudaMemcpy(m_ResourceBuffer, &data[0], byteSize, cudaMemcpyDefault);

		return true;
	}

	bool ResetBuffer()
	{
		//Requires Unified Virtual Addressing support : cudaMemcpyDefault
		uint32_t byteSize = GetTotalByteSize();
		switch (type)
		{
		case BufferType::GPU:
			GPUErrorCheck(cudaMemset(m_ResourceBuffer, 0, byteSize));
			break;
		case BufferType::CPU:
			memset(m_ResourceBuffer, 0, byteSize);
			break;
		}

		return true;
	}

	inline T* GetRawBuffer() const
	{ return m_ResourceBuffer; }

	inline BufferType GetType() const
	{ return m_Type; }

	inline uint32_t GetAmountElements() const
	{ return m_AmountElements; }

	inline uint32_t GetTotalByteSize() const
	{ return (sizeof(T) * m_AmountElements); }
};

template<typename T, BufferType typeSource, BufferType typeDestination>
bool CopyBuffers(GlobalResourceBuffer<T, typeDestination>* destination, const GlobalResourceBuffer<T, typeSource>* source)
{
	if (source->GetAmountElements() != destination->GetAmountElements())
		return false;

	T* const rawSourceBuffer = source->GetRawBuffer();
	T* const rawDestinationBuffer = destination->GetRawBuffer();
	uint32_t const byteSize = source->GetTotalByteSize();

	//Requires Unified Virtual Addressing support : cudaMemcpyDefault
	cudaMemcpy(rawDestinationBuffer, rawSourceBuffer, byteSize, cudaMemcpyDefault); 

	return true;
}