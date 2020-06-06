#include "Timer.h"

void Timer::Reset()
{
	auto currentTime = std::chrono::high_resolution_clock::now();

	m_StartTime = currentTime;
	m_PreviousTime = currentTime;

	m_FPSTimer = 0.0f;
	m_FPSCount = 0;
}

void Timer::Start()
{
	auto startTime = std::chrono::high_resolution_clock::now();
	m_StartTime = startTime;
	m_PreviousTime = startTime;
}

void Timer::Update()
{
	auto currentTime = std::chrono::high_resolution_clock::now();

	m_ElapsedTime = currentTime - m_PreviousTime;
	m_TotalTime = currentTime - m_StartTime;
	m_PreviousTime = currentTime;

	//FPS LOGIC
	m_FPSTimer += m_ElapsedTime.count() / 1000.0;
	++m_FPSCount;
	if (m_FPSTimer >= 1)
	{
		m_FPS = m_FPSCount;
		m_FPSCount = 0;
		m_FPSTimer -= 1;
	}
}
