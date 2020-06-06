/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Vector2.h: Vector2D struct
/*=============================================================================*/
#pragma once

#include "Vector.h"
#include "Point.h"
#include "MathUtilities.h"

//=== VECTOR2 SPECIALIZATION ===
template<typename T>
struct Vector<2, T>
{
	//=== Data ===
#pragma warning(disable : 4201)
	union
	{
		T data[2];
		struct { T x, y; };
		struct { T r, g; };
	};
#pragma warning(default : 4201)

	//=== Constructors ===
#pragma region Constructors
	BOTH_CALLABLE Vector<2, T>() : x(0), y(0) {};
	BOTH_CALLABLE Vector<2, T>(T _x, T _y)
		: x(_x), y(_y) {}
	BOTH_CALLABLE Vector<2, T>(const Vector<2, T>& v)
		: x(v.x), y(v.y) {}
	BOTH_CALLABLE Vector<2, T>(Vector<2, T>&& v) noexcept
		: x(std::move(v.x)), y(std::move(v.y)) {}
	BOTH_CALLABLE explicit Vector<2, T>(const Point<2, T>& p)
		: x(p.x), y(p.y) {}
#pragma endregion

	//=== Conversion Operator ===
#pragma region ConversionOperator
	template<typename U>
	BOTH_CALLABLE operator Vector<2, U>() const //Implicit conversion to different types of Vector2
	{
		return Vector<2, U>(
			static_cast<U>(this->x),
			static_cast<U>(this->y));
	}
#pragma endregion

	//=== Arithmetic Operators ===
#pragma region ArithmeticOperators
	template<typename U>
	BOTH_CALLABLE inline Vector<2, T> operator+(const Vector<2, U>& v) const
	{ return Vector<2, T>(x + static_cast<T>(v.x), y + static_cast<T>(v.y)); }

	template<typename U>
	BOTH_CALLABLE inline Vector<2, T> operator-(const Vector<2, U>& v) const
	{ return Vector<2, T>(x - static_cast<T>(v.x), y - static_cast<T>(v.y)); }

	BOTH_CALLABLE inline Vector<2, T> operator*(T scale) const
	{ return Vector<2, T>(x * scale, y * scale); }

	BOTH_CALLABLE inline Vector<2, T> operator/(T scale) const
	{
		const T revS = static_cast<T>(1.0f / scale);
		return Vector<2, T>(x * revS, y * revS);
	}
#pragma endregion

	//=== Compound Assignment Operators ===
#pragma region CompoundAssignmentOperators
	BOTH_CALLABLE inline Vector<2, T>& operator=(const Vector<2, T>& v)
	{ x = v.x; y = v.y; return *this; }

	BOTH_CALLABLE inline Vector<2, T>& operator+=(const Vector<2, T>& v)
	{ x += v.x; y += v.y; return *this; }

	BOTH_CALLABLE inline Vector<2, T>& operator-=(const Vector<2, T>& v)
	{ x -= v.x; y -= v.y; return *this;	}

	BOTH_CALLABLE inline Vector<2, T>& operator*=(T scale)
	{ x *= scale; y *= scale; return *this; }

	BOTH_CALLABLE inline Vector<2, T>& operator/=(T scale)
	{
		const T revS = static_cast<T>(1.0f / scale);
		x *= revS; y *= revS; return *this;
	}
#pragma endregion

	//=== Unary Operators ===
#pragma region UnaryOperators
	BOTH_CALLABLE inline Vector<2, T>& operator-() const
	{ return Vector<2, T>(-x, -y); }
#pragma endregion

	//=== Relational Operators ===
#pragma region RelationalOperators
	BOTH_CALLABLE inline bool operator==(const Vector<2, T>& v) const
	{ return AreEqual<T>(x, v.x) && AreEqual<T>(y, v.y); }

	BOTH_CALLABLE inline bool operator!=(const Vector<2, T>& v) const
	{ return !(*this == v); }
#pragma endregion 

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	BOTH_CALLABLE inline T operator[](uint8_t i) const
	{
		assert((i < 2) && "ERROR: index of Vector2 [] operator is out of bounds!");
		return data[i];
	}

	BOTH_CALLABLE inline T& operator[](uint8_t i)
	{
		assert((i < 2) && "ERROR: index of Vector2 [] operator is out of bounds!");
		return data[i];
	}
#pragma endregion

	//=== Static Functions ===
	BOTH_CALLABLE static Vector<2, T> ZeroVector();
};

//--- VECTOR3 FUNCTIONS ---
#pragma region GlobalOperators
template<typename T, typename U>
BOTH_CALLABLE inline Vector<2, T> operator*(U scale, const Vector<2, T>& v)
{
	T s = static_cast<T>(scale);
	return Vector<2, T>(v.x * s, v.y * s);
}
#pragma endregion

#pragma region GlobalFunctions
template<typename T>
BOTH_CALLABLE inline Vector<2, T> Vector<2, T>::ZeroVector()
{
	T z = static_cast<T>(0);
	return Vector<2, T>(z, z);
}

template<typename T>
BOTH_CALLABLE inline T Dot(const Vector<2, T>& v1, const Vector<2, T>& v2)
{ return v1.x * v2.x + v1.y * v2.y; }

//Returns 2D vector rotated 90 degrees counter-clockwise (where y-axis is up)
template<typename T>
BOTH_CALLABLE inline Vector<2, T> Perpendicular(const Vector<2, T>& v)
{ return Vector<2, T>(-v.y, v.x); }

template<typename T>
BOTH_CALLABLE inline Vector<2, T> GetAbs(const Vector<2, T>& v)
{ return Vector<2, T>(abs(v.x), abs(v.y)); }

template<typename T>
BOTH_CALLABLE inline Vector<2, T> Max(const Vector<2, T>& v1, const Vector<2, T>& v2)
{
	Vector<2, T>v = v1;
	if (v2.x > v.x) v.x = v2.x;
	if (v2.y > v.y) v.y = v2.y;
	return v;
}

template<typename T>
BOTH_CALLABLE inline Vector<2, T> Min(const Vector<2, T>& v1, const Vector<2, T>& v2)
{
	Vector<2, T>v = v1;
	if (v2.x < v.x) v.x = v2.x;
	if (v2.y < v.y) v.y = v2.y;
	return v;
}
#pragma endregion