/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// InputManager.h: manager class that controls the input in the engine.
/*=============================================================================*/
#pragma once
#include "Singleton.h"
#include "InputData.h"
#include <vector>
class SDLWindow;

/*! EInputManager: manager class that controls all the input, captured from active platform & window*/
class InputManager final : public Singleton<InputManager>
{
public:
	bool IsKeyboardKeyDown(InputScancode key) { return IsKeyPresent(InputType::eKeyboard, InputState::eDown, key); };
	bool IsKeyboardKeyUp(InputScancode key) { return IsKeyPresent(InputType::eKeyboard, InputState::eReleased, key); }

	bool IsMouseButtonDown(InputMouseButton button) { return IsMousePresent(InputType::eMouseButton, InputState::eDown, button); }
	bool IsMouseButtonUp(InputMouseButton button) { return IsMousePresent(InputType::eMouseButton, InputState::eReleased, button); }
	bool IsMouseScrolling() { return IsMousePresent(InputType::eMouseWheel); }
	bool IsMouseMoving() { return IsMousePresent(InputType::eMouseMotion); }
	MouseData GetMouseData(InputType type, InputMouseButton button = InputMouseButton(0));

	const uint8_t* GetKeyboardState() const
	{ return KeyboardState; }

	MouseState GetMouseState() const
	{ return CurrentMouseState; }

private:
	//=== Friends ===
	//Our window has access to add input events to our queue, our application can later use these events
	friend class Window;

	//=== Internal Functions
	void Flush(){ InputContainer.clear();};
	void AddInputAction(const InputAction& inputAction) 
	{ InputContainer.push_back(inputAction); };

	bool IsKeyPresent(InputType type, InputState state, InputScancode code);
	bool IsMousePresent(InputType type, InputState state = InputState(0), InputMouseButton button = InputMouseButton(0));

	void SetKeyboardState(const uint8_t* keyboardState)
	{ KeyboardState = keyboardState; }

	void SetMouseState(uint32_t mouseState, int x, int y)
	{ CurrentMouseState = MouseState(mouseState, x, y); }

	//=== Datamembers ===
	std::vector<InputAction> InputContainer = {};
	const uint8_t* KeyboardState = nullptr;
	MouseState CurrentMouseState = {};
};