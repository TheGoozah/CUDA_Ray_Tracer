/*=============================================================================*/
// Authors: Matthieu Delaere
/*=============================================================================*/
// Camera.h: class representing a camera
/*=============================================================================*/
#pragma once
#include "Math.h"
#include "GPUHelpers.h"
#include "InputManager.h"

struct CameraData
{
	FMatrix4 ONB = {};
	FPoint3 position = {};
	float fov = 0.f;
	float aspectRatio = 0.f;
};

struct Camera
{
	CPU_CALLABLE Camera(const FPoint3& _position, float fovAngle)
		:position(_position), FOVAngle(fovAngle)
	{}
	FPoint3 position = { 0.f, 0.f, 0.f };
	FVector3 u = { 1.f, 0.f, 0.f }; //ONB x-axis
	FVector3 v = { 0.f, 1.f, 0.f }; //ONB y-axis
	FVector3 w = { 0.f, 0.f, 1.f }; //ONB z-axis
	float FOVAngle = 90.f;
	bool didChange = false;

	CPU_CALLABLE void MoveForward(float d)
	{
		position += w * d;
		didChange = true;
	}

	CPU_CALLABLE void MoveRight(float d)
	{ 
		position += u * d;
		didChange = true;
	}

	CPU_CALLABLE void MoveUp(float d)
	{ 
		position += FVector3(0.f,1.f,0.f) * d;
		didChange = true;
	}

	CPU_CALLABLE void Pitch(float angle)
	{
		FMatrix3 rotation = MakeRotation(ToRadians(angle), u);
		w = Inverse(Transpose(rotation)) * w;
		didChange = true;
	}

	CPU_CALLABLE void Yaw(float angle)
	{
		FMatrix3 rotation = MakeRotation(ToRadians(angle), v);
		w = Inverse(Transpose(rotation)) * w;
		didChange = true;
	}

	CPU_CALLABLE FMatrix4 LookAt(const FVector3& up = FVector3(0.f, 1.f, 0.f))
	{
		//Compute the OrthoNormal Basis (U = local X, V = local Y, W = local Z)
		Normalize(w);
		u = Cross(up, w);
		Normalize(u);
		v = Cross(w, u);
		Normalize(v);

		FMatrix4 cameraToWorld = FMatrix4(
			FVector4(u), FVector4(v), FVector4(w), FVector4(FVector3(position), 1.f));
		return cameraToWorld;
	}

	CPU_CALLABLE void Update(float deltaTime)
	{
		//Key Movement
		InputManager* inputManager = InputManager::GetInstance();

		const uint8_t* pKeyboardState = inputManager->GetKeyboardState();
		//Key Movement
		if (pKeyboardState[(int)InputScancode::eScancode_S] || pKeyboardState[(int)InputScancode::eScancode_W]
			|| pKeyboardState[(int)InputScancode::eScancode_D] || pKeyboardState[(int)InputScancode::eScancode_A])
		{
			float cameraKeyboardSpeed = (pKeyboardState[(int)InputScancode::eScancode_LShift] + 1) * 20.f;
			MoveForward((pKeyboardState[(int)InputScancode::eScancode_S] - 
				pKeyboardState[(int)InputScancode::eScancode_W]) * cameraKeyboardSpeed * deltaTime);
			MoveRight((pKeyboardState[(int)InputScancode::eScancode_D] - 
				pKeyboardState[(int)InputScancode::eScancode_A]) * cameraKeyboardSpeed * deltaTime);
		}

		//Mouse Movement - UE4 like
		float cameraMouseSpeed = 4.0f;
		float rotationSensitivity = 0.075f;
		MouseState currentMouseState = inputManager->GetMouseState();
		if (currentMouseState.mouseState == InputMask((int)InputMouseButton::eLeft))
		{
			MoveForward(currentMouseState.y * cameraMouseSpeed * deltaTime);
			Yaw(currentMouseState.x * rotationSensitivity);
		}
		else if (currentMouseState.mouseState == InputMask((int)InputMouseButton::eRight))
		{
			Pitch(currentMouseState.y * rotationSensitivity);
			Yaw(currentMouseState.x * rotationSensitivity);
		}
		else if (currentMouseState.mouseState == (InputMask((int)InputMouseButton::eLeft) | InputMask((int)InputMouseButton::eRight)))
		{
			MoveUp(currentMouseState.y * cameraMouseSpeed * deltaTime);
		}
	}

	CPU_CALLABLE void Reset(const FPoint3& pos)
	{
		position = pos;
		u = { 1.f, 0.f, 0.f }; //ONB x-axis
		v = { 0.f, 1.f, 0.f }; //ONB y-axis
		w = { 0.f, 0.f, 1.f }; //ONB z-axis
		didChange = true;
	}
};