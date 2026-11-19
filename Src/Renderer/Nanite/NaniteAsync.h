#pragma once

#include "Renderer/Nanite/NaniteGpu.h"
#include <memory>
#include <unordered_map>

namespace Evans::Renderer
{
    inline constexpr unsigned NaniteAsyncMaximumTasks=16;
    inline constexpr std::size_t NaniteAsyncMaximumBytes=512ull*1024*1024;
    // A cancelled submission still occupies its budget until the device retires
    // it. This prevents rapid camera/source changes from accumulating hidden work.
    template<class Work,class Task=NaniteGeometryTask> class NaniteAsyncQueue
    {
        struct Entry { std::shared_ptr<Work> Data; bool Cancelled=false; };
        std::unordered_map<std::uint64_t,Entry> m_Entries;
    public:
        template<class Plan> bool CanSubmit(const Plan& plan,std::string& error) const
        {
            std::size_t bytes=0; for (const auto& entry:m_Entries) bytes+=entry.second.Data->Plan.ReservedBytes;
            if (m_Entries.size()>=NaniteAsyncMaximumTasks || plan.ReservedBytes>NaniteAsyncMaximumBytes-bytes)
            { error="Nanite async production exceeds its in-flight task or memory budget"; return false; }
            return true;
        }
        Task Insert(std::shared_ptr<Work> work)
        { Task task{NewNaniteGeometryTask().Value}; m_Entries.emplace(task.Value,Entry{std::move(work),false}); return task; }
        std::shared_ptr<Work> Find(Task task) const
        { const auto found=m_Entries.find(task.Value); return found==m_Entries.end() || found->second.Cancelled?nullptr:found->second.Data; }
        bool Cancel(Task task)
        { auto found=m_Entries.find(task.Value); if (found==m_Entries.end() || found->second.Cancelled) return false; found->second.Cancelled=true; return true; }
        void Erase(Task task) { m_Entries.erase(task.Value); }
        template<class Probe> bool Collect(Probe probe)
        {
            for (auto entry=m_Entries.begin();entry!=m_Entries.end();)
            {
                bool ready=false;
                if (entry->second.Cancelled && !probe(*entry->second.Data,ready)) return false;
                if (entry->second.Cancelled && ready) entry=m_Entries.erase(entry); else ++entry;
            }
            return true;
        }
        bool Empty() const { return m_Entries.empty(); }
        template<class Visitor> bool Visit(Visitor visitor)
        { for (auto& entry:m_Entries) if (!visitor(*entry.second.Data)) return false; return true; }
        void Clear() { m_Entries.clear(); } // Explicit queues are drained; implicit APIs retain submitted references.
    };
}
