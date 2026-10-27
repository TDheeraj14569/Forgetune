#include "forgetune/fusion.hpp"
#include <algorithm>

namespace forgetune {

bool OperatorFusionPass::run(IRGraph& graph) {
    bool any_changed = false;

    // We repeatedly look for fusion candidates until a fixed point is reached.
    bool changed_in_iteration = true;
    while (changed_in_iteration) {
        changed_in_iteration = false;

        auto use_counts = graph.use_counts();
        auto consumers = graph.consumer_map();
        const auto& graph_outputs = graph.get_outputs();

        // Search for candidate RELU nodes
        for (const auto& r_node : graph.nodes()) {
            if (r_node.deleted || r_node.op != OpKind::RELU || r_node.inputs.size() != 1) {
                continue;
            }

            NodeId in_id = r_node.inputs[0];
            if (!graph.has_node(in_id)) continue;
            const auto& in_node = graph.get_node(in_id);
            if (in_node.deleted) continue;

            // Pattern 1: CONV2D + RELU -> FUSED_CONV2D_RELU
            if (in_node.op == OpKind::CONV2D) {
                if (use_counts[in_id] == 1 &&
                    std::find(graph_outputs.begin(), graph_outputs.end(), in_id) == graph_outputs.end()) {

                    NodeId r_id = r_node.id;
                    std::string fused_name = "fused_" + in_node.name + "_" + r_node.name;
                    auto inputs = in_node.inputs;
                    auto output_meta = r_node.output_meta;
                    auto conv_attrs = in_node.conv_attrs;

                    IRNode fused_node;
                    fused_node.op = OpKind::FUSED_CONV2D_RELU;
                    fused_node.name = std::move(fused_name);
                    fused_node.inputs = std::move(inputs);
                    fused_node.output_meta = std::move(output_meta);
                    fused_node.conv_attrs = std::move(conv_attrs);

                    NodeId fused_id = graph.add_node(std::move(fused_node));
                    graph.replace_all_uses(r_id, fused_id);
                    graph.delete_node(r_id);
                    graph.delete_node(in_id);

                    changed_in_iteration = true;
                    any_changed = true;
                    break;
                }
            }

            // Pattern 2: MATMUL + ADD + RELU -> FUSED_MATMUL_ADD_RELU
            if (in_node.op != OpKind::ADD || in_node.inputs.size() != 2) {
                continue;
            }
            NodeId add_id = in_id;
            const auto& a_node = in_node;

            // Check if ADD node is safe to fuse:
            // Must have use count 1 (consumed only by this RELU), and must NOT be a graph output
            if (use_counts[add_id] != 1) continue;
            if (std::find(graph_outputs.begin(), graph_outputs.end(), add_id) != graph_outputs.end()) {
                continue;
            }

            // Find which input to ADD is MATMUL
            NodeId matmul_id = INVALID_NODE_ID;
            NodeId bias_id = INVALID_NODE_ID;

            if (graph.has_node(a_node.inputs[0]) && graph.get_node(a_node.inputs[0]).op == OpKind::MATMUL) {
                matmul_id = a_node.inputs[0];
                bias_id = a_node.inputs[1];
            } else if (graph.has_node(a_node.inputs[1]) && graph.get_node(a_node.inputs[1]).op == OpKind::MATMUL) {
                matmul_id = a_node.inputs[1];
                bias_id = a_node.inputs[0];
            } else {
                continue;
            }

            const auto& m_node = graph.get_node(matmul_id);
            if (m_node.deleted || m_node.inputs.size() != 2) continue;

            // Check if MATMUL node is safe to fuse:
            // Must have use count 1 (consumed only by this ADD), and must NOT be a graph output
            if (use_counts[matmul_id] != 1) continue;
            if (std::find(graph_outputs.begin(), graph_outputs.end(), matmul_id) != graph_outputs.end()) {
                continue;
            }

            // Candidate identified:
            // %M = MATMUL(%X, %W)
            // %A = ADD(%M, %B)
            // %R = RELU(%A)
            NodeId r_id = r_node.id;
            NodeId a_id = a_node.id;
            NodeId m_id = m_node.id;
            NodeId x_id = m_node.inputs[0];
            NodeId w_id = m_node.inputs[1];
            std::string fused_name = "fused_" + m_node.name + "_" + a_node.name + "_" + r_node.name;
            TensorMeta output_meta = r_node.output_meta;

            IRNode fused_node;
            fused_node.op = OpKind::FUSED_MATMUL_ADD_RELU;
            fused_node.name = std::move(fused_name);
            fused_node.inputs = {x_id, w_id, bias_id};
            fused_node.output_meta = std::move(output_meta);

            NodeId fused_id = graph.add_node(std::move(fused_node));

            // Replace all uses of RELU with the fused node
            graph.replace_all_uses(r_id, fused_id);

            // Mark nodes as deleted
            graph.delete_node(r_id);
            graph.delete_node(a_id);
            graph.delete_node(m_id);

            changed_in_iteration = true;
            any_changed = true;
            break; // Restart loop with fresh use-counts and consumer map
        }
    }

    return any_changed;
}

} // namespace forgetune
