/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Point.h: Point2D struct
/*=============================================================================*/
#pragma once

#include "Point.h"
#include "Vector.h"
#include "MathUtilities.h"

//=== POINT2 SPECIALIZATION ===
template<typename T>
struct Point<2, T>
{
	//=== Data ===
#pragma warning(disable : 4201)
	union
	{
		T data[2];
		struct { T x, y; };
	};
#pragma warning(default : 4201)

	//=== Constructors ===
#pragma region Constructors
	BOTH_CALLABLE Point<2, T>() : x(0), y(0) {};
	BOTH_CALLABLE Point<2, T>(T _x, T _y)
		: x(_x), y(_y) {}
	BOTH_CALLABLE Point<2, T>(const Point<2, T>& p)
		: x(p.x), y(p.y) {}
	BOTH_CALLABLE Point<2, T>(Point<2, T>&& p) noexcept
		:x(std::move(p.x)), y(std::move(p.y)) {}
	BOTH_CALLABLE explicit Point<2, T>(const Vector<2, T>& v)
		: x(v.x), y(v.y) {}
#pragma endregion

	//=== Conversion Operator ===
#pragma region ConversionOperator
	template<typename U>
	BOTH_CALLABLE operator Point<2, U>() const //Implicit conversion to different types of Point2
	{
		return Point<2, U>(
			static_cast<U>(this->x),
			static_cast<U>(this->y));
	}
#pragma endregion

	//=== Arithmetic Operators ===
#pragma region ArithmeticOperators
	template<typename U>
	BOTH_CALLABLE inline Point<2, T> operator+(const Vector<2, U>& v) const
	{ return Point<2, T>(x + static_cast<T>(v.x), y + static_cast<T>(v.y));	}

	template<typename U>
	BOTH_CALLABLE inline Point<2, T> operator-(const Vector<2, U>& v) const
	{ return Point<2, T>(x - static_cast<T>(v.x), y - static_cast<T>(v.y)); }

	template<typename U>
	BOTH_CALLABLE inline Vector<2, T> operator-(const Point<2, U>& p) const
	{ return Vector<2, T>(x - static_cast<T>(p.x), y - static_cast<T>(p.y)); }
#pragma endregion

	//=== Compound Assignment Operators ===
#pragma region CompoundAssignmentOperators
	BOTH_CALLABLE inline Point<2, T>& operator=(const Point<2, T>& p)
	{ x = p.x; y = p.y; return *this; }

	BOTH_CALLABLE inline Point<2, T>& operator+=(const Vector<2, T>& v)
	{ x += v.x; y += v.y; return *this; }

	BOTH_CALLABLE inline Point<2, T>& operator-=(const Vector<2, T>& v)
	{ x -= v.x; y -= v.y; return *this; }
#pragma endregion

	//=== Relational Operators ===
#pragma region RelationalOperators
	BOTH_CALLABLE inline bool operator==(const Point<2, T>& p) const
	{ return AreEqual<T>(x, p.x) && AreEqual<T>(y, p.y); }

	BOTH_CALLABLE inline bool operator!=(const Point<2, T>& p) const
	{ return !(*this == p); }
#pragma endregion 

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	BOTH_CALLABLE inline T operator[](uint8_t i) const
	{
		assert((i < 2) && "ERROR: index of Point2 [] operator is out of bounds!");
		return data[i];
	}

	BOTH_CALLABLE inline T& operator[](uint8_t i)
	{
		assert((i < 2) && "ERROR: index of Point2 [] operator is out of bounds!");
		return data[i];
	}
#pragma endregion
};