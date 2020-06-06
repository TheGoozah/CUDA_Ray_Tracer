#include "GPUHelpers.h"

class Spinlock final
{
	int* m_lock;

public:
	GPU_CALLABLE inline Spinlock(int* lock)
	{
		m_lock = lock;
	}

	//Disable copy constructor and assignment operator by not implementing them!
	GPU_CALLABLE Spinlock(const Spinlock&);
	GPU_CALLABLE Spinlock& operator=(const Spinlock&);

	GPU_CALLABLE inline void Lock()
	{
		while (atomicCAS(m_lock, 0, 1));
	}

	GPU_CALLABLE inline void Unlock()
	{
		atomicExch(m_lock, 0);
	}
};