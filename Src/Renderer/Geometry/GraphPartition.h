#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Evans::Renderer
{
    // Undirected CSR graph: strictly increasing neighbors per row, no self
    // edges, and matching positive weights in both directions. Empty weights
    // mean unit weights. Vertex IDs are local, zero-based indices.
    struct PartitionGraph
    {
        std::vector<int> Offsets;
        std::vector<int> Neighbors;
        std::vector<int> Weights;
    };

    struct GraphPartitionResult
    {
        std::vector<std::vector<std::uint32_t>> Parts;
        std::uint64_t CutWeight=0; // Each undirected cut edge is counted once.
    };

    // Project-local Riemann/METIS adaptation. Multilevel coarsening and
    // refinement minimize weighted edge cuts subject to approximate balance;
    // callers must enforce hard payload limits separately. Parts and members
    // are nonempty and canonically ordered. An empty graph has offsets {0}
    // and accepts zero parts. Failure leaves output unchanged.
    bool PartitionWeightedGraph(const PartitionGraph& graph,std::uint32_t partCount,
        GraphPartitionResult& output,std::string& error);
}
