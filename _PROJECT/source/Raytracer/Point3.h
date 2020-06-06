/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// EPoint.h: Point3D struct
/*=============================================================================*/
#pragma once

#include "Point.h"
#include "Vector.h"
#include "MathUtilities.h"

//=== POINT3 SPECIALIZATION ===
template<typename T>
struct Point<3, T>
{
	//=== Data ===
#pragma warning(disable : 4201)
	union
	{
		T data[3];
		struct { T x, y, z; };
		Point<2, T> xy;
	};
#pragma warning(default : 4201)

	//=== Constructors ===
#pragma region Constructors
	BOTH_CALLABLE Point<3, T>() : x(0), y(0), z(0) {};
	BOTH_CALLABLE Point<3, T>(T _x, T _y, T _z = 1)
		: x(_x), y(_y), z(_z) {}
	BOTH_CALLABLE Point<3, T>(const Point<3, T> & p)
		: x(p.x), y(p.y), z(p.z) {}
	BOTH_CALLABLE Point<3, T>(const Point<2, T> & p, T _z = 1)
		: x(p.x), y(p.y), z(_z) {}
	BOTH_CALLABLE Point<3, T>(Point<3, T> && p) noexcept
		:x(std::move(p.x)), y(std::move(p.y)), z(std::move(p.z)) {}
	BOTH_CALLABLE explicit Point<3, T>(const Vector<3, T> & v)
		: x(v.x), y(v.y), z(v.z) {}
	BOTH_CALLABLE explicit Point<3, T>(const Point<4, T> & p)
		: x(p.x), y(p.y), z(p.z) {}
#pragma endregion

	//=== Conversion Operator ===
#pragma region ConversionOperator
	template<typename U>
	BOTH_CALLABLE operator Point<3, U>() const //Implicit conversion to different types of Point3
	{
		return Point<3, U>(
			static_cast<U>(this->x),
			static_cast<U>(this->y),
			static_cast<U>(this->z));
	}
#pragma endregion

	//=== Arithmetic Operators ===
#pragma region ArithmeticOperators
	template<typename U>
	BOTH_CALLABLE inline Point<3, T> operator+(const Vector<3, U>& v) const
	{ return Point<3, T>(x + static_cast<T>(v.x), y + static_cast<T>(v.y), z + static_cast<T>(v.z)); }

	template<typename U>
	BOTH_CALLABLE inline Point<3, T> operator-(const Vector<3, U>& v) const
	{ return Point<3, T>(x - static_cast<T>(v.x), y - static_cast<T>(v.y), z - static_cast<T>(v.z)); }

	template<typename U>
	BOTH_CALLABLE inline Vector<3, T> operator-(const Point<3, U>& p) const
	{ return Vector<3, T>(x - static_cast<T>(p.x), y - static_cast<T>(p.y), z - static_cast<T>(p.z)); }
#pragma endregion

	//=== Compound Assignment Operators ===
#pragma region CompoundAssignmentOperators
	BOTH_CALLABLE inline Point<3, T>& operator=(const Point<3, T>& p)
	{ x = p.x; y = p.y; z = p.z; return *this; }

	BOTH_CALLABLE inline Point<3, T>& operator+=(const Vector<3, T>& v)
	{ x += v.x; y += v.y; z += v.z; return *this; }

	BOTH_CALLABLE inline Point<3, T>& operator-=(const Vector<3, T>& v)
	{ x -= v.x; y -= v.y; z -= v.z; return *this; }
#pragma endregion

	//=== Relational Operators ===
#pragma region RelationalOperators
	BOTH_CALLABLE inline bool operator==(const Point<3, T>& p) const
	{ return AreEqual<T>(x, p.x) && AreEqual<T>(y, p.y) && AreEqual<T>(z, p.z); }

	BOTH_CALLABLE inline bool operator!=(const Point<3, T>& p) const
	{ return !(*this == p); }
#pragma endregion 

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	BOTH_CALLABLE inline T operator[](uint8_t i) const
	{
		assert((i < 3) && "ERROR: index of Point3 [] operator is out of bounds!");
		return data[i];
	}

	BOTH_CALLABLE inline T& operator[](uint8_t i)
	{
		assert((i < 3) && "ERROR: index of Point3 [] operator is out of bounds!");
		return data[i];
	}
#pragma endregion
};