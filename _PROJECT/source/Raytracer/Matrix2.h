/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Matrix2.h: Column Major Matrix2x2 struct
/*=============================================================================*/
#pragma once

#include "Matrix.h"
#include "Vector.h"
#include "MathUtilities.h"

//=== MATRIX2x2 SPECIALIZATION ===
template<typename T>
struct Matrix<2, 2, T>
{		
	//=== Data ===
	T data[2][2];

	//=== Constructors ===
#pragma region Constructors
	BOTH_CALLABLE Matrix<2, 2, T>(){};
	//Every "row" of values passed here is a row in our matrix!
	BOTH_CALLABLE Matrix<2, 2, T>(T _00, T _01,
					T _10, T _11)
	{
		data[0][0] = _00; data[0][1] = _10;
		data[1][0] = _01; data[1][1] = _11;
	}
	//Every vector passed here is a column in our matrix!
	BOTH_CALLABLE Matrix<2, 2, T>(const Vector<2, T>& a, const Vector<2, T>& b)
	{
		data[0][0] = a.x; data[0][1] = a.y;
		data[1][0] = b.x; data[1][1] = b.y;
	}
	BOTH_CALLABLE Matrix<2, 2, T>(const Matrix<2, 2, T>& m)
	{
		data[0][0] = m.data[0][0]; data[0][1] = m.data[0][1];
		data[1][0] = m.data[1][0]; data[1][1] = m.data[1][1];
	}
	BOTH_CALLABLE Matrix<2, 2, T>(Matrix<2, 2, T>&& m) noexcept
	{
		data[0][0] = std::move(m.data[0][0]); data[0][1] = std::move(m.data[0][1]);
		data[1][0] = std::move(m.data[1][0]); data[1][1] = std::move(m.data[1][1]);
	}
#pragma endregion

	//=== Arithmetic Operators ===
#pragma region ArithmeticOperators
	BOTH_CALLABLE inline Matrix<2, 2, T> operator+(const Matrix<2, 2, T>& m) const
	{ 
		return Matrix<2, 2, T>(
			data[0][0] + m.data[0][0], data[0][1] + m.data[0][1],
			data[1][0] + m.data[1][0], data[1][1] + m.data[1][1]);
	}

	BOTH_CALLABLE inline Matrix<2, 2, T> operator-(const Matrix<2, 2, T>& m) const
	{ 
		return Matrix<2, 2, T>(
			data[0][0] - m.data[0][0], data[0][1] - m.data[0][1],
			data[1][0] - m.data[1][0], data[1][1] - m.data[1][1]);
	}

	template<typename U>
	BOTH_CALLABLE inline Matrix<2, 2, T> operator*(U scale) const
	{
		const T s = static_cast<T>(scale);
		return Matrix<2, 2, T>(
			data[0][0] * s, data[0][1] * s,
			data[1][0] * s, data[1][1] * s);
	}

	template<typename U>
	BOTH_CALLABLE inline Matrix<2, 2, T> operator/(U scale) const
	{
		const T revS = static_cast<T>(1.0f / scale);
		return Matrix<2, 2, T>(
			data[0][0] * revS, data[0][1] * revS,
			data[1][0] * revS, data[1][1] * revS);
	}

	BOTH_CALLABLE inline Matrix<2, 2, T> operator*(const Matrix<2, 2, T>& rm)
	{
		const Matrix<2, 2, T>& lm = (*this);
		return Matrix<2, 2, T>(
			lm(0, 0) * rm(0, 0) + lm(0, 1) * rm(1, 0),
			lm(0, 0) * rm(0, 1) + lm(0, 1) * rm(1, 1),
			
			lm(1, 0) * rm(0, 0) + lm(1, 1) * rm(1, 0),
			lm(1, 0) * rm(0, 1) + lm(1, 1) * rm(1, 1));
	}

	BOTH_CALLABLE inline Vector<2, T> operator*(const Vector<2, T>& v)
	{
		const Matrix<2, 2, T>& m = (*this);
		return Vector<2, T>(
			m(0, 0) * v.x + m(0, 1) * v.y,
			m(1, 0) * v.x + m(1, 1) * v.y);
	}
#pragma endregion

	//=== Compound Assignment Operators ===
#pragma region CompoundAssignmentOperators
	BOTH_CALLABLE inline Matrix<2, 2, T>& operator=(const Matrix<2, 2, T>& m)
	{
		data[0][0] = m.data[0][0]; data[0][1] = m.data[0][1];
		data[1][0] = m.data[1][0]; data[1][1] = m.data[1][1];
		return *this;
	}

	BOTH_CALLABLE inline Matrix<2, 2, T>& operator+=(const Matrix<2, 2, T>& m)
	{
		data[0][0] += m.data[0][0]; data[0][1] += m.data[0][1];
		data[1][0] += m.data[1][0]; data[1][1] += m.data[1][1];
		return *this;
	}

	BOTH_CALLABLE inline Matrix<2, 2, T>& operator-=(const Matrix<2, 2, T>& m)
	{
		data[0][0] -= m.data[0][0]; data[0][1] -= m.data[0][1];
		data[1][0] -= m.data[1][0]; data[1][1] -= m.data[1][1];
		return *this;
	}

	template<typename U>
	BOTH_CALLABLE inline Matrix<2, 2, T>& operator*=(U scale)
	{
		const T s = static_cast<T>(scale);
		data[0][0] *= s; data[0][1] *= s;
		data[1][0] *= s; data[1][1] *= s;
		return *this;
	}

	template<typename U>
	BOTH_CALLABLE inline Matrix<2, 2, T>& operator/=(U scale)
	{
		const T revS = static_cast<T>(1.0f / scale);
		data[0][0] *= revS; data[0][1] *= revS;
		data[1][0] *= revS; data[1][1] *= revS;
		return *this;
	}

	BOTH_CALLABLE inline Matrix<2, 2, T>& operator*=(const Matrix<2, 2, T>& m)
	{
		//Copy is necessary! :( 
		*this = *this * m;
		return *this;
	}
#pragma endregion

	//=== Relational Operators ===
#pragma region RelationalOperators
	BOTH_CALLABLE inline bool operator==(const Matrix<2, 2, T>& m) const
	{ return ((*this)[0] == m[0]) && ((*this)[1] == m[1]); }

	BOTH_CALLABLE inline bool operator!=(const Matrix<2, 2, T>& m) const
	{ return !(*this == m); }
#pragma endregion 

	//=== Member Access Operators ===
#pragma region MemberAccessOperators
	//Access parameter order still happens as row,column indexing (standard in programming)
	BOTH_CALLABLE inline T operator()(uint8_t r, uint8_t c) const
	{
		assert((r < 2 && c < 2) && "ERROR: indices of Matrix2x2 () const operator are out of bounds!");
		return (data[c][r]);
	}

	BOTH_CALLABLE inline T& operator()(uint8_t r, uint8_t c)
	{
		assert((r < 2 && c < 2) && "ERROR: indices of Matrix2x2 () operator are out of bounds!");
		return (data[c][r]);
	}

	//Get a vector representation of a column (usage), but internally returns one of the rows
	BOTH_CALLABLE inline const Vector<2, T>& operator[](uint8_t c) const
	{
		assert((c < 2) && "ERROR: index of Matrix2x2 [] operator is out of bounds!");
		return *((Vector<2, T>*)data[c]);
	}

	BOTH_CALLABLE inline Vector<2, T>& operator[](uint8_t c)
	{
		assert((c < 3) && "ERROR: index of Matrix2x2 [] operator is out of bounds!");
		return *((Vector<2, T>*)data[c]);
	}
#pragma endregion

	//=== Static Functions ===
	BOTH_CALLABLE static Matrix<2, 2, T> Identity();
};

//--- MATRIX2 FUNCTIONS ---
#pragma region GlobalFunctions
template<typename T>
BOTH_CALLABLE inline Matrix<2, 2, T> Matrix<2, 2, T>::Identity()
{
	return Matrix<2, 2, T>(
		1,0,
		0,1);
}

template<typename T>
BOTH_CALLABLE inline Matrix<2, 2, T> Transpose(const Matrix<2, 2, T>& m)
{
	Matrix<2, 2, T> t = {};
	t(0, 0) = m(0, 0);
	t(0, 1) = m(1, 0);
	t(1, 0) = m(0, 1);
	t(1, 1) = m(1, 1);
	return t;
}

template<typename T>
BOTH_CALLABLE inline T Determinant(const Matrix<2, 2, T>& m)
{ return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0); }

template<typename T>
BOTH_CALLABLE inline Matrix<2, 2, T> Inverse(const Matrix<2, 2, T>& m)
{
	T invDet = static_cast<T>(1.0f / Determinant(m));
	Matrix<2, 2, T> inv = {};
	inv(0, 0) = +m(1, 1) * invDet;
	inv(0, 1) = -m(0, 1) * invDet;
	inv(1, 0) = -m(1, 0) * invDet;
	inv(1, 1) = +m(0, 0) * invDet;
	return inv;
}

//Rotations with a positive angle are considered a counterclockwise rotations around the axis pointing towards
//the viewer.
template<typename T>
BOTH_CALLABLE inline Matrix<2, 2, T> MakeRotation(T t)
{
	T c = static_cast<T>(cos(t));
	T s = static_cast<T>(sin(t));

	return Matrix<2, 2, T>(
		c, -s,
		s, c);
}

template<typename T>
BOTH_CALLABLE inline Matrix<2, 2, T> MakeScale(T x, T y)
{
	return Matrix<2, 2, T>(
		x, static_cast<T>(0),
		static_cast<T>(0), y);
}
#pragma endregion