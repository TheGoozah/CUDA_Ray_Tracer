/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Math.h: General Math header that includes all math types and utilities
/*=============================================================================*/
#pragma once

/* --- STANDARD --- */
#include <math.h>
#include <cassert>
/* --- UTILITIES --- */
#include "MathUtilities.h"
/* --- TYPES --- */
#include "Vector.h"
#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Point.h"
#include "Point2.h"
#include "Point3.h"
#include "Point4.h"
#include "Matrix.h"
#include "Matrix2.h"
#include "Matrix3.h"
#include "Matrix4.h"

/* --- TYPE DEFINES --- */
typedef Vector<2, float>	FVector2;
typedef Vector<2, double>	DVector2;
typedef Vector<2, int>		IVector2;
typedef Vector<3, float>	FVector3;
typedef Vector<3, double>	DVector3;
typedef Vector<3, int>		IVector3;
typedef Vector<4, float>	FVector4;
typedef Vector<4, double>	DVector4;
typedef Vector<4, int>		IVector4;

typedef Point<2, float>		FPoint2;
typedef Point<2, double>	DPoint2;
typedef Point<2, int>		IPoint2;
typedef Point<3, float>		FPoint3;
typedef Point<3, double>	DPoint3;
typedef Point<3, int>		IPoint3;
typedef Point<4, float>		FPoint4;
typedef Point<4, double>	DPoint4;
typedef Point<4, int>		IPoint4;

typedef Matrix<2, 2, float>		FMatrix2;
typedef Matrix<2, 2, double>	DMatrix2;
typedef Matrix<3, 3, float>		FMatrix3;
typedef Matrix<3, 3, double>	DMatrix3;
typedef Matrix<4, 4, float>		FMatrix4;
typedef Matrix<4, 4, double>	DMatrix4;
