#pragma once
#include <assert.h>
#include "GPUHelpers.h"

namespace cc //CUDA Containers
{
	template<typename T>
	class vector final
	{
		T* m_Data = nullptr;
		uint32_t m_Capacity = 0;
		uint32_t m_AmountElements = 0;
		bool m_Destroyed = false;

		GPU_CALLABLE inline void resize(uint32_t newCapacity)
		{
			if (newCapacity <= m_Capacity)
			{
				if (m_Destroyed)
					newCapacity = 1;
				else
					return;
			}

			T* newData = new T[newCapacity];
			memcpy(newData, m_Data, sizeof(T) * m_Capacity);
			delete[] m_Data;
			m_Data = newData;
			m_Capacity = newCapacity;
			m_Destroyed = false;
		}

	public:
		GPU_CALLABLE vector(uint32_t capacity = 1) : 
			m_Capacity(capacity)
		{
			m_Data = new T[m_Capacity];
		};
		GPU_CALLABLE ~vector()
		{
			if(m_Data)
				delete[] m_Data;
		};

		GPU_CALLABLE vector(const vector& other) : 
			vector(other.m_Capacity)
		{
			m_AmountElements = other.m_AmountElements;
			memcpy(m_Data, other.m_Data, sizeof(T) * m_Capacity);
		}
		GPU_CALLABLE vector& operator=(const vector& other) 
		{
			if (this != &other)
			{
				delete[] m_Data;
				m_AmountElements = other.m_AmountElements;
				m_Capacity = other.m_Capacity;
				m_Data = new T[m_Capacity];
				memcpy(m_Data, other.m_Data, sizeof(T) * m_Capacity);
			}
			return *this;
		}

		GPU_CALLABLE vector(const vector&& other):
			m_Data(std::move(other.m_Data)), 
			m_Capacity(std::move(other.m_Capacity)), m_AmountElements(std::move(other.m_AmountElements))
		{}
		GPU_CALLABLE vector& operator=(const vector&& other) 
		{
			if (this != &other)
			{
				delete[] m_Data;
				m_Data = std::move(other.m_Data);
				m_Capacity = std::move(other.m_Capacity);
				m_AmountElements = std::move(other.m_AmountElements);
			}
			return *this;
		}

		GPU_CALLABLE T operator[](int index) const
		{
			assert(index < m_AmountElements);
			return *(m_Data + index);
		}
		GPU_CALLABLE T& operator[](int index)
		{
			assert(index < m_AmountElements);
			return *(m_Data + index);
		}

		GPU_CALLABLE inline bool empty() const
		{ return (m_AmountElements == 0); }
		GPU_CALLABLE inline uint32_t size() const
		{ return m_AmountElements; }
		GPU_CALLABLE inline uint32_t capacity() const
		{ return m_Capacity; }

		GPU_CALLABLE void push_back(const T& value)
		{
			if (m_AmountElements >= m_Capacity)
			{		
				uint32_t newCapacity = m_Capacity * 2;
				resize(newCapacity);
			}

			T* newElement = m_Data + m_AmountElements;
			*newElement = value;
			++m_AmountElements;
		};
		GPU_CALLABLE void push_back(const T&& value)
		{
			if (m_AmountElements >= m_Capacity)
			{
				uint32_t newCapacity = m_Capacity * 2;
				resize(newCapacity);
			}
			T* newElement = m_Data + m_AmountElements;
			*newElement = std::move(value);
			++m_AmountElements;
		}

		GPU_CALLABLE inline T at(uint32_t index) const
		{
			assert(index < m_AmountElements);
			return *(m_Data + index);
		}
		GPU_CALLABLE inline T* data() const
		{ return m_Data; }

		GPU_CALLABLE inline void destroy()
		{
			for (uint32_t i = 0; i < m_AmountElements; ++i)
			{
				T obj = at(i);
				if (obj)
					delete obj;
			}
			if(m_Data)
				delete[] m_Data;
			m_AmountElements = 0;
			m_Capacity = 0;
			m_Destroyed = true;
		}
	};
}