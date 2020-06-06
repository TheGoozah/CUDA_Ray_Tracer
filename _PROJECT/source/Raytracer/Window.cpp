#include "Window.h"
#include "InputManager.h"

#include <utility>
#include <iostream>

Window::Window(WindowSettings&& settings):
	m_WindowSettings(std::move(settings)), m_Window(nullptr), 
	m_FrontBufferSurface(nullptr), m_BackBufferSurface(nullptr),
	m_BackBuffer(nullptr), m_ShutdownRequested(false)
{}

bool Window::CreateBuffers(SDL_Window* const window)
{
	m_FrontBufferSurface = std::unique_ptr<SDL_Surface, std::function<void(SDL_Surface*)>>(
		SDL_GetWindowSurface(window), [](SDL_Surface*) {}); //Empty deleter to avoid deletion by the window. SDL owns the surface!
	m_BackBufferSurface = std::unique_ptr<SDL_Surface, std::function<void(SDL_Surface*)>>(
		SDL_CreateRGBSurface(0, m_WindowSettings.width, m_WindowSettings.height, 32, 0, 0, 0, 0),
		[](SDL_Surface* surface)
		{
			SDL_FreeSurface(surface);
		});
	m_BackBuffer = std::make_unique<uint32_t[]>((uint64_t)m_WindowSettings.width * m_WindowSettings.height);

	return false;
}

void Window::TakeScreenshot()
{
	SDL_SaveBMP(m_BackBufferSurface.get(), "BackBufferRender.bmp");
}

bool Window::CreateWindow()
{
	uint32_t flags = 0;
	if (m_WindowSettings.fullScreen)
		flags |= SDL_WINDOW_FULLSCREEN;

	SDL_Init(SDL_INIT_VIDEO);
	m_Window = std::unique_ptr<SDL_Window, std::function<void(SDL_Window*)>>(
		SDL_CreateWindow("CUDA Harware Accelerated Raytracer",
			SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
			m_WindowSettings.width, m_WindowSettings.height, flags),
		[](SDL_Window* window) 
		{
			SDL_DestroyWindow(window); 
		});

	if (m_Window != nullptr)
	{
		CreateBuffers(m_Window.get());
		return true;
	}

	return false;
}

void Window::PollEvents()
{
	InputManager::GetInstance()->Flush();

	//Mouse & Keyboard STATES
	int x, y = 0;
	uint32_t mouseState = SDL_GetRelativeMouseState(&x, &y);
	InputManager::GetInstance()->SetMouseState(mouseState, x, y);
	InputManager::GetInstance()->SetKeyboardState(SDL_GetKeyboardState(0));

	//Mouse & Keyboard EVENTS
	SDL_Event e;
	while (SDL_PollEvent(&e))
	{
		switch (e.type)
		{
		case SDL_QUIT:
			m_ShutdownRequested = true;
			break;
		case SDL_KEYDOWN:
		{
			auto data = KeyboardData(
				static_cast<int>(e.key.timestamp),
				static_cast<InputScancode>(e.key.keysym.scancode));

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eKeyboard, InputState::eDown, InputData(data)));
			break;
		}
		case SDL_KEYUP:
		{
			auto data = KeyboardData(
				static_cast<int>(e.key.timestamp),
				static_cast<InputScancode>(e.key.keysym.scancode));

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eKeyboard, InputState::eReleased, InputData(data)));
			break;
		}
		case SDL_MOUSEBUTTONDOWN:
		{
			//SDL_GetMouseState -> Relative to Desktop, not Window
			int x, y;
			SDL_GetMouseState(&x, &y);

			auto data = MouseData(
				static_cast<int>(e.key.timestamp),
				static_cast<InputMouseButton>(e.button.button),
				x, y);

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eMouseButton, InputState::eDown, InputData(data)));
			break;
		}
		case SDL_MOUSEBUTTONUP:
		{
			int x, y;
			SDL_GetMouseState(&x, &y);

			auto data = MouseData(
				static_cast<int>(e.key.timestamp),
				static_cast<InputMouseButton>(e.button.button),
				x, y);

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eMouseButton, InputState::eReleased, InputData(data)));
			break;
		}
		case SDL_MOUSEWHEEL:
		{
			auto data = MouseData(
				static_cast<int>(e.key.timestamp),
				InputMouseButton(0), e.wheel.x, e.wheel.y, 0, 0);

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eMouseWheel, InputState(0), InputData(data)));
			break;
		}
		case SDL_MOUSEMOTION:
		{
			int x, y;
			SDL_GetMouseState(&x, &y);

			auto data = MouseData(
				static_cast<int>(e.key.timestamp),
				InputMouseButton(0), x, y, e.motion.xrel, e.motion.yrel);

			InputManager::GetInstance()->AddInputAction(InputAction(
				InputType::eMouseMotion, InputState(0), InputData(data)));
			break;
		}
		}
	}
}

void Window::Present()
{
	SDL_Surface* backBufferSurface = m_BackBufferSurface.get();

	//TODO: check if we can just share the pixel buffer with the GPU VM so we don't need to do this extra copy.
	//Be careful though, the surface needs to be locked, so before GPU invocation, if you would do this.
	SDL_LockSurface(backBufferSurface);
	memcpy(backBufferSurface->pixels, m_BackBuffer.get(), 
		sizeof(uint32_t) * m_WindowSettings.width * m_WindowSettings.height);
	SDL_UnlockSurface(backBufferSurface);

	SDL_BlitSurface(backBufferSurface, 0, m_FrontBufferSurface.get(), 0);
	SDL_UpdateWindowSurface(m_Window.get());
}
