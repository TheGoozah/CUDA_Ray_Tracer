#pragma once

#pragma warning(push)
#pragma warning(disable: 26812)
#include <SDL.h>
#include <SDL_surface.h>
#pragma warning(pop)

#include <memory>
#include <functional>

struct WindowSettings final
{
	uint32_t width = 0;
	uint32_t height = 0;
	bool fullScreen = false;

	WindowSettings() {};
	WindowSettings(uint32_t const _width, uint32_t const _height, bool _fullScreen) :
		width(_width), height(_height), fullScreen(_fullScreen)
	{}
};

class Window final
{
	WindowSettings const m_WindowSettings;
	std::unique_ptr<SDL_Window, std::function<void(SDL_Window*)>> m_Window;

	std::unique_ptr<SDL_Surface, std::function<void(SDL_Surface*)>> m_FrontBufferSurface;
	std::unique_ptr<SDL_Surface, std::function<void(SDL_Surface*)>> m_BackBufferSurface;
	std::unique_ptr<uint32_t[]> m_BackBuffer;

	bool m_ShutdownRequested;

	bool CreateBuffers(SDL_Window* const window);

public:
	Window(WindowSettings&& settings);
	~Window() = default;
	
	bool CreateWindow();
	void PollEvents();
	void Present();
	void TakeScreenshot();

	const WindowSettings& GetWindowSettings() const
	{ return m_WindowSettings; }

	WindowSettings GetWindowSettingsCopy() const
	{ return m_WindowSettings; }

	uint32_t* GetBackbufferPixels() const
	{ return m_BackBuffer.get(); }

	bool ShutdownRequested() const { return m_ShutdownRequested; };
};

