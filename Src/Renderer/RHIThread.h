#pragma once

#include "Renderer.h"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

namespace Evans::Renderer
{
	// The main thread only produces commands. Renderer and RHI calls are consumed
	// serially by the dedicated RHI thread after Start has been called.
	class RHIThread
	{
	public:
		using Command = std::function<void(Renderer&)>;

		explicit RHIThread(Renderer& renderer);
		~RHIThread();

		RHIThread(const RHIThread&) = delete;
		RHIThread& operator=(const RHIThread&) = delete;

		bool Start();
		void Stop(bool drainPendingCommands = true);
		bool Enqueue(Command command);
		std::future<bool> EnqueueFrame();
		void Flush();
		bool IsRunning() const;
		bool IsOnRHIThread() const;
		std::size_t GetPendingCommandCount() const;

	private:
		void ThreadMain();

		Renderer& m_Renderer;
		mutable std::mutex m_Mutex;
		std::condition_variable m_Condition;
		std::deque<Command> m_Commands;
		std::thread m_Thread;
		std::thread::id m_RHIThreadId;
		bool m_Running = false;
		bool m_StopRequested = false;
		bool m_DrainOnStop = true;
	};
}
