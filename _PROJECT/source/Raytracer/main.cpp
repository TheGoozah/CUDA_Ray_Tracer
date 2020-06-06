#include "Window.h"
#undef main //SDL2 defines main to be SDL_main...
#include "InputManager.h"

#include "Tracer.cuh"
#include "Buffers.h"

#include "Timer.h"
#include "DeviceContext.cuh"

int main()
{
	//Create window and loop while no shutdown requested
	std::unique_ptr<Window> window = std::make_unique<Window>(WindowSettings(640, 480, false));
	bool succeeded = window->CreateWindow();

	//--- Rendering Buffers ---
	const WindowSettings& windowSettings = window->GetWindowSettings();
	auto accumulationBuffer = std::make_unique<GlobalResourceBuffer<RGBColor, BufferType::GPU>>(windowSettings.width * windowSettings.height);
	auto pixelBuffer = std::make_unique<GlobalResourceBuffer<uint32_t, BufferType::GPU>>(windowSettings.width * windowSettings.height);
	auto backBuffer = std::make_unique<GlobalResourceBuffer<uint32_t, BufferType::CPU>>(
		window->GetBackbufferPixels(), windowSettings.width * windowSettings.height);

	//Timer
	std::unique_ptr<Timer> timer = std::make_unique<Timer>(Timer());
	timer->Start();

	//Create camera
	FPoint3 cameraStartPosition = FPoint3(0.f, 2.f, 8.f);
	auto cpuCamera = std::make_unique<Camera>(cameraStartPosition, 45.f);

	//Create Device Context
	auto deviceContext = DeviceContext(windowSettings);

	//Debug parameters
	RenderParameters renderParams = {};

	//INFO
	std::cout 
		<< "----------------------------------------------" << std::endl
		<< "Move camera: WSAD" << std::endl
		<< "Rotate camera: Right MouseButton" << std::endl
		<< "Fly through camera: Left MouseButton Drag" << std::endl
		<< "Up/Down camera: Right + Left MouseButton Drag" << std::endl
		<< "Enable/Disable Indirect Lighting (1 bounce): E" << std::endl
		<< "Reset camera: R" << std::endl
		<< "Take screenshot: T" << std::endl
		<< "----------------------------------------------" << std::endl;

	bool shutdownRequested = false;
	double printInSecondsTimer = 0.0;
	while (!shutdownRequested)
	{
		//Input Events
		window->PollEvents();

		//Change parameters based on input
		if (InputManager::GetInstance()->IsKeyboardKeyUp(InputScancode::eScancode_E))
		{
			if (renderParams.amountBouncesPerHit == 1)
				++renderParams.amountBouncesPerHit;
			else if (renderParams.amountBouncesPerHit == 2)
				--renderParams.amountBouncesPerHit;

			accumulationBuffer.get()->ResetBuffer();
		}

		if (InputManager::GetInstance()->IsKeyboardKeyUp(InputScancode::eScancode_T))
			window->TakeScreenshot();

		if (InputManager::GetInstance()->IsKeyboardKeyUp(InputScancode::eScancode_R))
			cpuCamera->Reset(cameraStartPosition);

		//Update camera
		float deltaTime = (float)timer->GetElapsedSeconds();
		cpuCamera->Update(deltaTime);
		
		//Update device context
		bool validRenderState = deviceContext.Update(windowSettings, cpuCamera.get(), renderParams);

		//Reset buffers if necessary
		if (cpuCamera.get()->didChange)
			accumulationBuffer.get()->ResetBuffer();
		cpuCamera.get()->didChange = false;

		if (validRenderState)
		{
			//Do rendering...
			Render(accumulationBuffer->GetRawBuffer(), pixelBuffer->GetRawBuffer(), windowSettings.width, windowSettings.height, deviceContext);
			//Copy Data
			CopyBuffers(backBuffer.get(), pixelBuffer.get());
			//Present
			window->Present();
		}

		//Performance timing
		timer->Update();
		printInSecondsTimer += timer->GetElapsedSeconds();
		if (printInSecondsTimer >= 3.f)
		{
			printInSecondsTimer = 0.0;
			std::cout << "Total ms: " << timer->GetElapsedMilliseconds()
				<< ", stable FPS: " << timer->GetFPS() << ", Elapsed FPS: " << timer->GetElapsedFPS() <<std::endl;
		}

		shutdownRequested = window->ShutdownRequested();
	}

	//Destroy managers
	InputManager::GetInstance()->Destroy();

    return 0;
}
