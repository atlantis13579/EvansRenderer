#include "Renderer/RHIThread.h"

#include <memory>
#include <utility>

namespace Evans::Renderer
{
	RHIThread::RHIThread(Renderer& renderer)
		: m_Renderer(renderer)
	{
	}

	RHIThread::~RHIThread()
	{
		Stop(true);
	}

	bool RHIThread::Start()
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		if (m_Running)
		{
			return false;
		}
		m_StopRequested = false;
		m_DrainOnStop = true;
		m_Running = true;
		m_Thread = std::thread(&RHIThread::ThreadMain, this);
		return true;
	}

	void RHIThread::Stop(bool drainPendingCommands)
	{
		{
			std::lock_guard<std::mutex> lock(m_Mutex);
			if (!m_Running)
			{
				return;
			}
			m_StopRequested = true;
			m_DrainOnStop = drainPendingCommands;
			if (!drainPendingCommands)
			{
				m_Commands.clear();
			}
		}
		m_Condition.notify_all();
		if (m_Thread.joinable())
		{
			m_Thread.join();
		}
		std::lock_guard<std::mutex> lock(m_Mutex);
		m_Running = false;
		m_RHIThreadId = {};
	}

	bool RHIThread::Enqueue(Command command)
	{
		if (!command)
		{
			return false;
		}
		{
			std::lock_guard<std::mutex> lock(m_Mutex);
			if (!m_Running || m_StopRequested)
			{
				return false;
			}
			m_Commands.push_back(std::move(command));
		}
		m_Condition.notify_one();
		return true;
	}

	std::future<bool> RHIThread::EnqueueFrame()
	{
		auto result = std::make_shared<std::promise<bool>>();
		std::future<bool> future = result->get_future();
		if (!Enqueue([result](Renderer& renderer) {
			bool succeeded = false;
			try
			{
				succeeded = renderer.RenderFrame();
			}
			catch (...)
			{
				// The producer receives a failed frame instead of an unresolved future.
			}
			result->set_value(succeeded);
		}))
		{
			result->set_value(false);
		}
		return future;
	}

	void RHIThread::Flush()
	{
		if (IsOnRHIThread())
		{
			return;
		}
		auto barrier = std::make_shared<std::promise<void>>();
		std::future<void> completed = barrier->get_future();
		if (Enqueue([barrier](Renderer&) { barrier->set_value(); }))
		{
			completed.wait();
		}
	}

	bool RHIThread::IsRunning() const
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		return m_Running && !m_StopRequested;
	}

	bool RHIThread::IsOnRHIThread() const
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		return m_Running && std::this_thread::get_id() == m_RHIThreadId;
	}

	std::size_t RHIThread::GetPendingCommandCount() const
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		return m_Commands.size();
	}

	void RHIThread::ThreadMain()
	{
		{
			std::lock_guard<std::mutex> lock(m_Mutex);
			m_RHIThreadId = std::this_thread::get_id();
		}

		while (true)
		{
			Command command;
			{
				std::unique_lock<std::mutex> lock(m_Mutex);
				m_Condition.wait(lock, [this]() { return m_StopRequested || !m_Commands.empty(); });
				if (m_StopRequested && (!m_DrainOnStop || m_Commands.empty()))
				{
					break;
				}
				if (!m_Commands.empty())
				{
					command = std::move(m_Commands.front());
					m_Commands.pop_front();
				}
			}

			if (command)
			{
				try
				{
					command(m_Renderer);
				}
				catch (...)
				{
					// A bad producer command must not terminate the process or the RHI thread.
				}
			}
		}
	}
}
