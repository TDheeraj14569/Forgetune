#include "forgetune/memory_planner.hpp"
#include "forgetune/graph_verifier.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstdint>
#include <stdexcept>

namespace forgetune {

// Maximum defensible padding dimension for convolution scratch buffer allocation
constexpr int64_t kMaxConvPadding = 100000;

std::string MemoryPlan::to_string() const {
    std::ostringstream oss;
    oss << "MemoryPlan:\n"
        << "  Arena Size:                     " << arena_size_bytes << " bytes\n"
        << "  Total Intermediate (Without):   " << naive_size_bytes << " bytes\n"
        << "  Peak Concurrent Live Storage:   " << peak_live_bytes << " bytes\n"
        << "  Conv2D Scratch Workspace:       " << conv2d_scratch_bytes << " bytes\n"
        << "  Savings (Arena vs No-Reuse):    " << std::fixed << std::setprecision(2) << savings_percent << "%\n"
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
    auto v_res = verify_graph(graph, false);
    if (!v_res.passed) {
        std::string err = v_res.errors.empty() ? "unknown verification error" : v_res.errors[0];
        throw std::runtime_error("MemoryPlanner: graph verification failed: " + err);
    }

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

    // 4. Calculate maximum concurrent live intermediate storage across execution steps
    size_t peak_live = 0;
    for (int step = 0; step < total_steps; ++step) {
        size_t step_live = 0;
        for (const auto& alloc : candidates) {
            if (alloc.birth_step <= step && step <= alloc.death_step) {
                step_live += alloc.size_bytes;
            }
        }
        peak_live = std::max(peak_live, step_live);
    }
    plan.peak_live_bytes = peak_live;

    // 5. Calculate Conv2D padding scratch workspace requirements across distinct generated Conv2D helpers
    size_t total_conv2d_scratch = 0;
    for (NodeId nid : order) {
        const auto& node = graph.get_node(nid);
        if (node.deleted) continue;
        if (node.op == OpKind::CONV2D || node.op == OpKind::FUSED_CONV2D_RELU) {
            int64_t ph = node.conv_attrs.pad_h;
            int64_t pw = node.conv_attrs.pad_w;
            if (ph > 0 || pw > 0) {
                if (node.inputs.size() >= 2 && graph.has_node(node.inputs[0])) {
                    const auto& in_node = graph.get_node(node.inputs[0]);
                    if (in_node.output_meta.shape.rank() == 4) {
                        int64_t C_in = in_node.output_meta.shape.dims[1];
                        int64_t H_in = in_node.output_meta.shape.dims[2];
                        int64_t W_in = in_node.output_meta.shape.dims[3];
                        if (ph < 0 || pw < 0 || ph > kMaxConvPadding || pw > kMaxConvPadding) {
                            throw std::runtime_error("MemoryPlanner: invalid conv padding");
                        }
                        if (H_in > INT64_MAX - 2 * ph || W_in > INT64_MAX - 2 * pw) {
                            throw std::overflow_error("MemoryPlanner: conv geometry overflow");
                        }
                        int64_t H_pad = H_in + 2 * ph;
                        int64_t W_pad = W_in + 2 * pw;
                        if (C_in <= 0 || H_pad <= 0 || W_pad <= 0) {
                            throw std::runtime_error("MemoryPlanner: non-positive conv geometry");
                        }
                        if (C_in > INT64_MAX / H_pad) throw std::overflow_error("Conv2D scratch overflow");
                        int64_t ch = C_in * H_pad;
                        if (ch > INT64_MAX / W_pad) throw std::overflow_error("Conv2D scratch overflow");
                        int64_t pad_elems = ch * W_pad;
                        if (pad_elems > static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
                            throw std::overflow_error("Conv2D scratch overflow");
                        }
                        size_t scratch_bytes = static_cast<size_t>(pad_elems) * sizeof(float);
                        if (std::numeric_limits<size_t>::max() - total_conv2d_scratch < scratch_bytes) {
                            throw std::overflow_error("Conv2D scratch total overflow");
                        }
                        total_conv2d_scratch += scratch_bytes;
                    }
                }
            }
        }
    }
    plan.conv2d_scratch_bytes = total_conv2d_scratch;

    if (plan.naive_size_bytes > 0) {
        plan.savings_percent = 100.0 * (1.0 - static_cast<double>(plan.arena_size_bytes) /
                                              static_cast<double>(plan.naive_size_bytes));
    }

    return plan;
}

} // namespace forgetune
