/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Timer.h: timer to get FPS
/*=============================================================================*/
#pragma once
#include <cstdint>
#include <chrono>

class Timer final
{
public:
	void Reset();
	void Start();
	void Update();

	uint32_t inline GetFPS() const { return m_FPS; };
	uint32_t inline GetElapsedFPS() const { return static_cast<uint32_t>(1.0 / (GetElapsedSeconds())); }
	double inline GetElapsedSeconds() const { return m_ElapsedTime.count() / 1000.0; };
	double inline GetElapsedMilliseconds() const { return m_ElapsedTime.count(); };
	double inline GetTotal() const { return m_TotalTime.count(); };

private:
	//std::chrono::steady_clock::time_point m_BaseTime;
	std::chrono::high_resolution_clock::time_point m_StartTime = {};
	std::chrono::high_resolution_clock::time_point m_PreviousTime = {};

	uint32_t m_FPS = 0;
	uint32_t m_FPSCount = 0;

	std::chrono::duration<double, std::milli> m_TotalTime = {};
	std::chrono::duration<double, std::milli> m_ElapsedTime = {};
	double m_FPSTimer = 0.0f;
};