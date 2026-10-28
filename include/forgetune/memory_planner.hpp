#ifndef FORGETUNE_MEMORY_PLANNER_HPP
#define FORGETUNE_MEMORY_PLANNER_HPP

#include "forgetune/ir.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

namespace forgetune {

struct BufferAllocation {
    NodeId node_id = INVALID_NODE_ID;
    size_t offset = 0;
    size_t size_bytes = 0;
    int birth_step = 0;
    int death_step = 0;
};

struct MemoryPlan {
    size_t arena_size_bytes = 0;
    size_t naive_size_bytes = 0;
    double savings_percent = 0.0;
    std::unordered_map<NodeId, BufferAllocation> allocations;

    std::string to_string() const;
};

class MemoryPlanner {
public:
    explicit MemoryPlanner(size_t alignment = 64) : alignment_(alignment) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("MemoryPlanner: alignment must be a non-zero power of 2");
        }
    }

    // Computes liveness intervals and assigns buffer offsets in an arena
    MemoryPlan plan(const IRGraph& graph, int64_t max_batch = TensorShape::DEFAULT_MAX_BATCH) const;

private:
    size_t alignment_;
    size_t align_up(size_t val) const {
        if (val > (std::numeric_limits<size_t>::max() - (alignment_ - 1))) {
            throw std::overflow_error("MemoryPlanner::align_up arithmetic overflow");
        }
        return ((val + alignment_ - 1) / alignment_) * alignment_;
    }
};

} // namespace forgetune

#endif // FORGETUNE_MEMORY_PLANNER_HPP
