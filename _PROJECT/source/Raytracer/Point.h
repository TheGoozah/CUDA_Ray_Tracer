/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Point.h: base point struct + general point template functions
/*=============================================================================*/
#pragma once
#include "GPUHelpers.h"

//=== CORE POINT TYPE ===
template<int N, typename T>
struct Point
{
	//=== Data ===
	T data[N];

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	BOTH_CALLABLE inline T operator[](uint8_t i) const
	{
		assert((i < N) && "ERROR: index of Vector [] operator is out of bounds!");
		return data[i];
	}

	BOTH_CALLABLE inline T& operator[](uint8_t i)
	{
		assert((i < N) && "ERROR: index of Vector [] operator is out of bounds!");
		return data[i];
	}
#pragma endregion
};

template<int N, typename T>
BOTH_CALLABLE inline T SqrDistance(const Point<N, T>& p1, const Point<N, T>& p2)
{
	const Vector<N, T> diff = p2 - p1;
	return SqrMagnitude(diff);
}

template<int N, typename T>
BOTH_CALLABLE inline T Distance(const Point<N, T>& p1, const Point<N, T>& p2)
{ return static_cast<T>(sqrt(SqrDistance(p1, p2))); }