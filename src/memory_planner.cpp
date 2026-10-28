#include "forgetune/memory_planner.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>

namespace forgetune {

std::string MemoryPlan::to_string() const {
    std::ostringstream oss;
    oss << "MemoryPlan:\n"
        << "  Arena Size: " << arena_size_bytes << " bytes\n"
        << "  Naive Peak: " << naive_size_bytes << " bytes\n"
        << "  Savings:    " << std::fixed << std::setprecision(2) << savings_percent << "%\n"
        << "  Allocations:\n";
    for (const auto& kv : allocations) {
        const auto& alloc = kv.second;
        oss << "    Node %" << alloc.node_id << ": offset " << alloc.offset
            << ", size " << alloc.size_bytes << " bytes (lifespan ["
            << alloc.birth_step << ", " << alloc.death_step << "])\n";
    }
    return oss.str();
}

MemoryPlan MemoryPlanner::plan(const IRGraph& graph, int64_t max_batch) const {
    MemoryPlan plan;
    auto order = graph.topological_order();
    int total_steps = static_cast<int>(order.size());

    // 1. Map each node to its execution step
    std::unordered_map<NodeId, int> step_map;
    for (int step = 0; step < total_steps; ++step) {
        step_map[order[static_cast<size_t>(step)]] = step;
    }

    const auto& graph_outputs = graph.get_outputs();
    std::unordered_set<NodeId> output_set(graph_outputs.begin(), graph_outputs.end());

    // Build consumer list for each node
    std::unordered_map<NodeId, std::vector<NodeId>> consumers;
    for (NodeId nid : order) {
        const auto& node = graph.get_node(nid);
        if (node.deleted) continue;
        for (NodeId in_id : node.inputs) {
            consumers[in_id].push_back(nid);
        }
    }

    // 2. Identify intermediate buffers and determine liveness intervals
    std::vector<BufferAllocation> candidates;
    for (NodeId nid : order) {
        const auto& node = graph.get_node(nid);
        if (node.deleted || node.op == OpKind::INPUT || node.op == OpKind::CONST) {
            continue;
        }
        if (output_set.find(nid) != output_set.end()) {
            continue; // Graph output is written to user-supplied buffer, not intermediate arena
        }

        BufferAllocation alloc;
        alloc.node_id = nid;
        alloc.birth_step = step_map[nid];

        // Death step is the latest consumer step, or total_steps if graph output
        int death = alloc.birth_step;
        if (output_set.find(nid) != output_set.end()) {
            death = total_steps;
        }
        auto it = consumers.find(nid);
        if (it != consumers.end()) {
            for (NodeId c_id : it->second) {
                if (step_map.find(c_id) != step_map.end()) {
                    death = std::max(death, step_map[c_id]);
                }
            }
        }
        alloc.death_step = death;

        int64_t count = node.output_meta.shape.element_count(max_batch);
        if (count < 0 || count > static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float)) || count > (INT64_MAX / 4)) {
            throw std::overflow_error("MemoryPlanner: tensor element count exceeds maximum representable memory size");
        }
        size_t raw_bytes = static_cast<size_t>(count) * sizeof(float);
        alloc.size_bytes = align_up(raw_bytes);
        if (std::numeric_limits<size_t>::max() - plan.naive_size_bytes < alloc.size_bytes) {
            throw std::overflow_error("MemoryPlanner: naive_size_bytes arithmetic overflow");
        }
        candidates.push_back(alloc);
        plan.naive_size_bytes += alloc.size_bytes;
    }

    // 3. Greedy best-fit interval coloring
    // Placed allocations
    std::vector<BufferAllocation> placed;

    for (auto& alloc : candidates) {
        // Collect all candidate offsets: 0 and end offsets of overlapping placed buffers
        std::vector<size_t> candidate_offsets;
        candidate_offsets.push_back(0);

        for (const auto& p : placed) {
            // Check if lifespans overlap
            bool lifespans_overlap = !(alloc.death_step < p.birth_step || p.death_step < alloc.birth_step);
            if (lifespans_overlap) {
                if (std::numeric_limits<size_t>::max() - p.offset < p.size_bytes) {
                    throw std::overflow_error("MemoryPlanner: buffer offset arithmetic overflow");
                }
                candidate_offsets.push_back(p.offset + p.size_bytes);
            }
        }

        std::sort(candidate_offsets.begin(), candidate_offsets.end());
        candidate_offsets.erase(std::unique(candidate_offsets.begin(), candidate_offsets.end()), candidate_offsets.end());

        // Find the lowest candidate offset that does not conflict with any overlapping placed buffer
        size_t best_offset = 0;
        bool found = false;

        for (size_t cand : candidate_offsets) {
            size_t cand_aligned = align_up(cand);
            bool collision = false;

            for (const auto& p : placed) {
                bool lifespans_overlap = !(alloc.death_step < p.birth_step || p.death_step < alloc.birth_step);
                if (lifespans_overlap) {
                    size_t start1 = cand_aligned;
                    if (std::numeric_limits<size_t>::max() - cand_aligned < alloc.size_bytes) {
                        collision = true;
                        break;
                    }
                    size_t end1 = cand_aligned + alloc.size_bytes;
                    size_t start2 = p.offset;
                    size_t end2 = p.offset + p.size_bytes;

                    if (std::max(start1, start2) < std::min(end1, end2)) {
                        collision = true;
                        break;
                    }
                }
            }

            if (!collision) {
                best_offset = cand_aligned;
                found = true;
                break;
            }
        }

        if (!found) {
            best_offset = align_up(plan.arena_size_bytes);
        }

        if (std::numeric_limits<size_t>::max() - best_offset < alloc.size_bytes) {
            throw std::overflow_error("MemoryPlanner: arena_size_bytes arithmetic overflow");
        }
        alloc.offset = best_offset;
        placed.push_back(alloc);
        plan.allocations[alloc.node_id] = alloc;
        plan.arena_size_bytes = std::max(plan.arena_size_bytes, alloc.offset + alloc.size_bytes);
    }

    if (plan.naive_size_bytes > 0) {
        plan.savings_percent = 100.0 * (1.0 - static_cast<double>(plan.arena_size_bytes) /
                                              static_cast<double>(plan.naive_size_bytes));
    }

    return plan;
}

} // namespace forgetune
