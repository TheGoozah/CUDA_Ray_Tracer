/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Vector3.h: Vector3D struct
/*=============================================================================*/
#pragma once

#include "Vector.h"
#include "Point.h"
#include "MathUtilities.h"

//=== VECTOR3 SPECIALIZATION ===
template<typename T>
struct Vector<3, T>
{
	//=== Data ===
#pragma warning(disable : 4201)
	union
	{
		T data[3];
		struct { T x, y, z; };
		struct { T r, g, b; };
		Vector<2, T> xy;
		Vector<2, T> rg;
	};
#pragma warning(default : 4201)

	//=== Constructors ===
#pragma region Constructors
	BOTH_CALLABLE Vector<3, T>() : x(0), y(0), z(0) {};
	BOTH_CALLABLE Vector<3, T>(T _x, T _y, T _z = 0)
		: x(_x), y(_y), z(_z) {}
	BOTH_CALLABLE Vector<3, T>(const Vector<3, T> & v)
		: x(v.x), y(v.y), z(v.z) {}
	BOTH_CALLABLE Vector<3, T>(const Vector<2, T> & v, T _z = 0)
		: x(v.x), y(v.y), z(_z) {}
	BOTH_CALLABLE Vector<3, T>(Vector<3, T> && v) noexcept
		:x(std::move(v.x)), y(std::move(v.y)), z(std::move(v.z)) {}
	BOTH_CALLABLE explicit Vector<3, T>(const Point<3, T> & p)
		: x(p.x), y(p.y), z(p.z) {}
	BOTH_CALLABLE explicit Vector<3, T>(const Vector<4, T> & v)
		: x(v.x), y(v.y), z(v.z) {}
#pragma endregion

	//=== Conversion Operator ===
#pragma region ConversionOperator
	template<typename U>
	BOTH_CALLABLE operator Vector<3, U>() const //Implicit conversion to different types of Vector3
	{
		return Vector<3, U>(
			static_cast<U>(this->x),
			static_cast<U>(this->y),
			static_cast<U>(this->z));
	}
#pragma endregion

	//=== Arithmetic Operators ===
#pragma region ArithmeticOperators
	template<typename U>
	BOTH_CALLABLE inline Vector<3, T> operator+(const Vector<3, U>& v) const
	{ return Vector<3, T>(x + static_cast<T>(v.x), y + static_cast<T>(v.y), z + static_cast<T>(v.z)); }

	template<typename U>
	BOTH_CALLABLE inline Vector<3, T> operator-(const Vector<3, U>& v) const
	{ return Vector<3, T>(x - static_cast<T>(v.x), y - static_cast<T>(v.y), z - static_cast<T>(v.z)); }

	BOTH_CALLABLE inline Vector<3, T> operator*(T scale) const
	{ return Vector<3, T>(x * scale, y * scale, z * scale); }

	BOTH_CALLABLE inline Vector<3, T> operator/(T scale) const
	{
		const T revS = static_cast<T>(1.0f / scale);
		return Vector<3, T>(x * revS, y * revS, z * revS);
	}
#pragma endregion

	//=== Compound Assignment Operators ===
#pragma region CompoundAssignmentOperators
	BOTH_CALLABLE inline Vector<3, T>& operator=(const Vector<3, T>& v)
	{ x = v.x; y = v.y; z = v.z; return *this; }

	BOTH_CALLABLE inline Vector<3, T>& operator+=(const Vector<3, T>& v)
	{ x += v.x; y += v.y; z += v.z; return *this; }

	BOTH_CALLABLE inline Vector<3, T>& operator-=(const Vector<3, T>& v)
	{ x -= v.x; y -= v.y; z -= v.z; return *this; }

	BOTH_CALLABLE inline Vector<3, T>& operator*=(T scale)
	{ x *= scale; y *= scale; z *= scale; return *this; }

	BOTH_CALLABLE inline Vector<3, T>& operator/=(T scale)
	{
		const T revS = static_cast<T>(1.0f / scale);
		x *= revS; y *= revS; z *= revS; return *this;
	}
#pragma endregion

	//=== Unary Operators ===
#pragma region UnaryOperators
	BOTH_CALLABLE inline Vector<3, T> operator-() const
	{ return Vector<3, T>(-x, -y, -z); }
#pragma endregion

	//=== Relational Operators ===
#pragma region RelationalOperators
	BOTH_CALLABLE inline bool operator==(const Vector<3, T>& v) const
	{ return AreEqual<T>(x, v.x) && AreEqual<T>(y, v.y) && AreEqual<T>(z, v.z); }

	BOTH_CALLABLE inline bool operator!=(const Vector<3, T>& v) const
	{ return !(*this == v); }
#pragma endregion 

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	BOTH_CALLABLE inline T operator[](uint8_t i) const
	{
		assert((i < 3) && "ERROR: index of Vector3 [] operator is out of bounds!");
		return data[i];
	}

	BOTH_CALLABLE inline T& operator[](uint8_t i)
	{
		assert((i < 3) && "ERROR: index of Vector3 [] operator is out of bounds!");
		return data[i];
	}
#pragma endregion

	//=== Static Functions ===
	BOTH_CALLABLE static Vector<3, T> ZeroVector();
};

//--- VECTOR3 FUNCTIONS ---
#pragma region GlobalOperators
template<typename T, typename U>
BOTH_CALLABLE inline Vector<3, T> operator*(U scale, const Vector<3, T>& v)
{
	T s = static_cast<T>(scale);
	return Vector<3, T>(v.x * s, v.y * s, v.z * s);
}
#pragma endregion

#pragma region GlobalFunctions
template<typename T>
BOTH_CALLABLE inline Vector<3, T> Vector<3, T>::ZeroVector()
{ 
	T z = static_cast<T>(0);
	return Vector<3, T>(z, z, z); 
}

template<typename T>
BOTH_CALLABLE inline T Dot(const Vector<3, T>& v1, const Vector<3, T>& v2)
{ return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

template<typename T>
BOTH_CALLABLE inline Vector<3, T> Cross(const Vector<3, T>& v1, const Vector<3, T>& v2)
{
	return Vector<3, T>{
		v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x};
}

template<typename T>
BOTH_CALLABLE inline Vector<3, T> GetAbs(const Vector<3, T>& v)
{ return Vector<3, T>(abs(v.x), abs(v.y), abs(v.z)); }

template<typename T>
BOTH_CALLABLE inline Vector<3, T> Max(const Vector<3, T>& v1, const Vector<3, T>& v2)
{
	Vector<3, T>v = v1;
	if (v2.x > v.x) v.x = v2.x;
	if (v2.y > v.y) v.y = v2.y;
	if (v2.z > v.z) v.z = v2.z;
	return v;
}

template<typename T>
BOTH_CALLABLE inline Vector<3, T> Min(const Vector<3, T>& v1, const Vector<3, T>& v2)
{
	Vector<3, T>v = v1;
	if (v2.x < v.x) v.x = v2.x;
	if (v2.y < v.y) v.y = v2.y;
	if (v2.z < v.z) v.z = v2.z;
	return v;
}
#pragma endregion