#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace Evans::Renderer
{
    struct LumenUpdateJob
    {
        std::size_t ReservedBytes=0;
        unsigned SubmittedUnits=0;
        bool Initialized=false,InFlight=false,Cancelled=false,Failed=false;
    };
    // Cancelled submissions retain their full memory reservation until native
    // completion. Publication follows the final retired slice, never submission.
    template<class Work,class Task> class LumenUpdateQueue
    {
        static constexpr unsigned MaximumTasks=4;
        static constexpr std::size_t MaximumBytes=512ull*1024*1024;
        std::unordered_map<std::uint64_t,std::shared_ptr<Work>> Entries;
    public:
        bool CanSubmit(const Work& work,std::string& error) const
        {
            std::size_t reserved=0; for (const auto& entry:Entries) reserved+=entry.second->ReservedBytes;
            if (Entries.size()>=MaximumTasks || reserved>MaximumBytes || work.ReservedBytes>MaximumBytes-reserved)
            { error="Lumen updates exceed their in-flight memory or task budget"; return false; }
            return true;
        }
        Task Insert(std::shared_ptr<Work> work)
        {
            const Task task{Work::NewTaskValue()};
            if (task) Entries.emplace(task.Value,std::move(work));
            return task;
        }
        bool Cancel(Task task)
        {
            const auto found=Entries.find(task.Value);
            if (found==Entries.end() || found->second->Cancelled) return false;
            found->second->Cancelled=true; return true;
        }
        template<class Probe> bool Collect(Probe probe)
        {
            for (auto entry=Entries.begin();entry!=Entries.end();)
            {
                auto& work=*entry->second; bool retired=!work.InFlight;
                if (work.Cancelled && !retired && !probe(work,retired)) return false;
                if (work.Cancelled && retired) entry=Entries.erase(entry); else ++entry;
            }
            return true;
        }
        template<class Output,class Statistics,class Probe,class Submit,class Publish>
        bool Advance(Task task,unsigned budget,bool& ready,Output& output,
            Statistics& statistics,std::string& error,Probe probe,Submit submit,Publish publish)
        {
            ready=false; const auto found=Entries.find(task.Value);
            if (found==Entries.end() || found->second->Cancelled || found->second->Failed)
            { error="Invalid or failed Lumen update task"; return false; }
            auto& work=*found->second;
            if (work.InFlight)
            {
                bool complete=false;
                if (!probe(work,complete)) { work.Failed=true; return false; }
                if (!complete) { statistics=work.Statistics; return true; }
                work.InFlight=false; work.RetireUnits(work.SubmittedUnits);
            }
            const unsigned remaining=work.RemainingUnits();
            if (work.Initialized && !remaining)
            {
                Output candidate;
                if (!publish(work,candidate)) { work.Failed=true; return false; }
                statistics=work.Statistics; Entries.erase(found); output=candidate; ready=true; return true;
            }
            auto frame=work.Slice(budget);
            if (!submit(work,frame)) { work.Failed=true; return false; }
            work.Initialized=true; work.SubmittedUnits=frame.Dispatch.Y;
            work.Submitted(frame.Dispatch.Y); statistics=work.Statistics; return true;
        }
        bool Empty() const { return Entries.empty(); }
        template<class Visitor> bool Visit(Visitor visitor)
        { for (auto& entry:Entries) if (!visitor(*entry.second)) return false; return true; }
        void Clear() { Entries.clear(); } // Native owners drain submitted commands before destruction.
    };
}
