//=== General Includes ===
#include "InputManager.h"
#include <algorithm>

//=== Public Functions ===
MouseData InputManager::GetMouseData(InputType type, InputMouseButton button)
{
	auto result = std::find_if(InputContainer.begin(), InputContainer.end(),
		[type, button](const InputAction& ia)
	{
		return
			(ia.inputActionType == type) &&
			(ia.inputActionData.mouseInputData.button == button);
	});

	if (result != InputContainer.end())
		return (*result).inputActionData.mouseInputData;
	return
		MouseData();
}

//=== Private Functions ===
bool InputManager::IsKeyPresent(InputType type, InputState state, InputScancode code)
{
	auto result = std::find_if(InputContainer.begin(), InputContainer.end(),
		[type, state, code](const InputAction& ia)
	{
		return
			(ia.inputActionType == type) &&
			(ia.inputActionState == state) &&
			(ia.inputActionData.keyboardInputData.scanCode == code);
	});
	return (result != InputContainer.end());
}

bool InputManager::IsMousePresent(InputType type, InputState state, InputMouseButton button)
{
	auto result = std::find_if(InputContainer.begin(), InputContainer.end(),
		[type, state, button](const InputAction& ia)
	{
		return
			(ia.inputActionType == type) &&
			(ia.inputActionState == state) &&
			(ia.inputActionData.mouseInputData.button == button);
	});
	return (result != InputContainer.end());
}