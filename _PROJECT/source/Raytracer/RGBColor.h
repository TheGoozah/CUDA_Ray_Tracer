/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// RGBColor.h: struct that represents a RGB color
/*=============================================================================*/
#pragma once

#include "GPUHelpers.h"
#include "MathUtilities.h"
#include <algorithm>

struct RGBColor final
{
	//=== Datamembers ===
	float r = 0.f;
	float g = 0.f;
	float b = 0.f;

	//=== Constructors & Destructor ===
	BOTH_CALLABLE RGBColor() {};
	BOTH_CALLABLE RGBColor(float _r, float _g, float _b) :r(_r), g(_g), b(_b) {}
	BOTH_CALLABLE RGBColor(const RGBColor& c) : r(c.r), g(c.g), b(c.b) {}
	BOTH_CALLABLE RGBColor(RGBColor&& c) noexcept : r(std::move(c.r)), g(std::move(c.g)), b(std::move(c.b)) {}
	BOTH_CALLABLE ~RGBColor() {};

	//=== Operators ===
	BOTH_CALLABLE RGBColor& operator=(const RGBColor& c)
	{ r = c.r; g = c.g; b = c.b; return *this; }
	BOTH_CALLABLE RGBColor& operator=(RGBColor&& c) noexcept
	{ r = std::move(c.r); g = std::move(c.g); b = std::move(c.b); return *this;	}

	//=== Arithmetic Operators ===
	BOTH_CALLABLE inline RGBColor operator+(const RGBColor& c) const
	{ return RGBColor(r + c.r, g + c.g, b + c.b); }
	BOTH_CALLABLE inline RGBColor operator-(const RGBColor& c) const
	{ return RGBColor(r - c.r, g - c.g, b - c.b); }
	BOTH_CALLABLE inline RGBColor operator*(const RGBColor& c) const
	{ return RGBColor(r * c.r, g * c.g, b * c.b); }
	BOTH_CALLABLE inline RGBColor operator/(float f) const
	{
		float rev = 1.0f / f;
		return RGBColor(r * rev, g * rev, b * rev);
	}
	BOTH_CALLABLE inline RGBColor operator*(float f) const
	{ return RGBColor(r * f, g * f, b * f);	}
	BOTH_CALLABLE inline RGBColor operator/(const RGBColor& c) const
	{ return RGBColor(r / c.r, g / c.g, b / c.b); }

	//=== Compound Assignment Operators ===
	BOTH_CALLABLE inline RGBColor& operator+=(const RGBColor& c)
	{ r += c.r; g += c.g; b += c.b; return *this; }
	BOTH_CALLABLE inline RGBColor& operator-=(const RGBColor& c)
	{ r -= c.r; g -= c.g; b -= c.b; return *this; }
	BOTH_CALLABLE inline RGBColor& operator*=(const RGBColor& c)
	{ r *= c.r; g *= c.g; b *= c.b; return *this; }
	BOTH_CALLABLE inline RGBColor& operator/=(const RGBColor& c)
	{ r /= c.r; g /= c.g; b /= c.b; return *this; }
	BOTH_CALLABLE inline RGBColor& operator*=(float f)
	{ r *= f; g *= f; b *= f; return *this; }
	BOTH_CALLABLE inline RGBColor& operator/=(float f)
	{
		float rev = 1.0f / f;
		r *= rev; g *= rev; b *= rev; return *this;
	}

	//=== Internal RGBColor Functions ===
	BOTH_CALLABLE inline void ClampColor()
	{
		r = Clamp(r, 0.0f, 1.0f);
		g = Clamp(g, 0.0f, 1.0f);
		b = Clamp(b, 0.0f, 1.0f);
	}

	BOTH_CALLABLE inline void MaxToOne()
	{
		float maxValue = std::max(r, std::max(g, b));
		if (maxValue > 1.f)
			*this /= maxValue;
	}
};

//=== Global RGBColor Functions ===
BOTH_CALLABLE inline RGBColor Max(const RGBColor& c1, const RGBColor& c2)
{
	RGBColor c = c1;
	if (c2.r > c.r) c.r = c2.r;
	if (c2.g > c.g) c.g = c2.g;
	if (c2.b > c.b) c.b = c2.b;
	return c;
}

BOTH_CALLABLE inline RGBColor Min(const RGBColor& c1, const RGBColor& c2)
{
	RGBColor c = c1;
	if (c2.r < c.r) c.r = c2.r;
	if (c2.g < c.g) c.g = c2.g;
	if (c2.b < c.b) c.b = c2.b;
	return c;
}

BOTH_CALLABLE inline uint32_t GetSDL_ARGBColor(const RGBColor& c)
{
	RGBColor rsColor = c * 255;
	uint32_t finalColor = 0;
	finalColor |= (uint8_t)rsColor.b;
	finalColor |= (uint8_t)rsColor.g << 8;
	finalColor |= (uint8_t)rsColor.r << 16;
	return finalColor;
}

BOTH_CALLABLE inline RGBColor GetColorFromSDL_ARGB(const uint32_t c)
{
	RGBColor color =
	{
		float(uint8_t(c >> 16))/255.f,
		float(uint8_t(c >> 8)) / 255.f,
		float(uint8_t(c)) / 255.f
	};
	return color;
}

BOTH_CALLABLE inline RGBColor GammaCorrection(const RGBColor& c)
{
	RGBColor result = c;
	float gamma = 1 / 2.2f;
	result.r = std::powf(result.r, gamma);
	result.g = std::powf(result.g, gamma);
	result.b = std::powf(result.b, gamma);
	result.MaxToOne();
	return result;
}

BOTH_CALLABLE inline RGBColor GammaCorrectionSRGB(const RGBColor& c)
{
	RGBColor result = c;
	if (result.r <= 0.0031308f)
		result.r *= 12.92f;
	else
		result.r = 1.055f * powf(result.r, 1 / 2.4f) - 0.055f;
	if (result.g <= 0.0031308f)
		result.g *= 12.92f;
	else
		result.g = 1.055f * powf(result.g, 1 / 2.4f) - 0.055f;
	if (result.b <= 0.0031308f)
		result.b *= 12.92f;
	else
		result.b = 1.055f * powf(result.b, 1 / 2.4f) - 0.055f;
	result.MaxToOne();
	return result;
}

BOTH_CALLABLE inline RGBColor ReinhardToneMapping(const RGBColor& c)
{
	//https://en.wikipedia.org/wiki/Tone_mapping
	return c / (c + RGBColor(1, 1, 1));
}

BOTH_CALLABLE inline RGBColor ACESFilmToneMapping(const RGBColor& color)
{
	//https://github.com/CJT-Jackton/RayTracing/wiki/Tone-mapping
	const float a = 2.51f;
	const float b = 0.03f;
	const float c = 2.43f;
	const float d = 0.59f;
	const float e = 0.14f;
	RGBColor result = (color * ((color * a) + RGBColor(b, b, b))) /
		(color * ((color * c) + RGBColor(d, d, d)) + RGBColor(e, e, e));
	result.ClampColor();
	return result;
}