/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// InputData.h: all structs/data used by the input manager
/*=============================================================================*/
#pragma once
#include "InputCodes.h"

struct KeyboardData //8 bytes
{
	int timeStamp = 0;
	InputScancode scanCode = InputScancode::eScancode_Unknown;

	KeyboardData(int _timeStamp, InputScancode keyCode) :
		timeStamp(_timeStamp), scanCode(keyCode)
	{}
};

struct MouseData //== 24 bytes
{
	int timeStamp = 0;
	InputMouseButton button = InputMouseButton(0);
	int x = 0; //Position X relative to window OR amound of scroll, based on Type!
	int y = 0; //Position Y relative to window OR amound of scroll, based on Type!
	int xRel = 0;
	int yRel = 0; //Y == Direction when scrolling (1 == UP, -1 == DOWN)

	MouseData() {};
	MouseData(int _timeStamp, InputMouseButton _button, int _x, int _y,
		int _xRel = 0, int _yRel = 0) :
		timeStamp(_timeStamp), button(_button), x(_x), y(_y),
		xRel(_xRel), yRel(_yRel)
	{}
};

struct MouseState
{
	int mouseState = 0;
	int x = 0;
	int y = 0;

	MouseState() {};
	MouseState(int _state, int _x, int _y):
		mouseState(_state), x(_x), y(_y)
	{}
};

union InputData //"Wasting" 16 bytes for a more user-friendly setup, SDL wastes even more memory (= 48 bytes)
{
	MouseData mouseInputData;
	KeyboardData keyboardInputData;

	InputData(MouseData data) : mouseInputData(data) {}
	InputData(KeyboardData data) : keyboardInputData(data) {}
};

//=== Actual InputAction used by the InputManager ===
struct InputAction
{
	InputType inputActionType = InputType::eDefault;
	InputState inputActionState = InputState::eDown;
	InputData inputActionData;

	InputAction(InputType type, InputState state, InputData data) :
		inputActionType(type), inputActionState(state), inputActionData(data) {}
};