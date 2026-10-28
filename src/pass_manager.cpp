#include "forgetune/pass_manager.hpp"
#include "forgetune/graph_verifier.hpp"
#include "forgetune/interpreter.hpp"
#include <algorithm>
#include <unordered_set>

namespace forgetune {

void PassManager::add_pass(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
}

std::vector<PassResult> PassManager::run(IRGraph& graph) {
    std::vector<PassResult> results;

    // Verify initial graph before any pass runs
    verify_graph(graph, true);

    for (auto& pass : passes_) {
        PassResult pr;
        pr.pass_name = pass->name();
        if (capture_ir_) {
            pr.ir_before = graph.to_string();
        }

        try {
            pr.changed = pass->run(graph);
            pr.succeeded = true;
            pr.message = pr.changed ? "Pass modified graph." : "Pass made no changes.";

            // Verify graph after every pass that made changes
            if (pr.changed) {
                verify_graph(graph, true);
            }
        } catch (const std::exception& e) {
            pr.succeeded = false;
            pr.message = std::string("Pass failed with error: ") + e.what();
            if (capture_ir_) {
                pr.ir_after = graph.to_string();
            }
            results.push_back(pr);
            throw; // Re-throw to fail closed
        }

        if (capture_ir_) {
            pr.ir_after = graph.to_string();
        }
        results.push_back(pr);
    }

    return results;
}

// ============================================================================
// Pass A: Constant Folding
// ============================================================================
bool ConstantFoldingPass::run(IRGraph& graph) {
    bool any_changed = false;
    Interpreter interp;

    // Iterate through live nodes in topological order
    auto order = graph.topological_order();
    for (NodeId id : order) {
        if (!graph.has_node(id)) continue;
        auto& node = graph.get_node_mut(id);
        if (node.deleted || node.op == OpKind::CONST || node.op == OpKind::INPUT || node.op == OpKind::OUTPUT) {
            continue;
        }

        // Check if all operands are CONST nodes
        bool all_const = true;
        for (NodeId in_id : node.inputs) {
            if (!graph.has_node(in_id) || graph.get_node(in_id).op != OpKind::CONST) {
                all_const = false;
                break;
            }
        }

        if (all_const && !node.inputs.empty()) {
            // Fold this node!
            std::vector<float> folded_data;
            if (node.op == OpKind::MATMUL) {
                const auto& c0 = graph.get_node(node.inputs[0]);
                const auto& c1 = graph.get_node(node.inputs[1]);
                // Build a mini-subgraph or evaluate directly via interpreter math
                // Using interpreter helper directly:
                // Since eval_matmul is private to interpreter, we evaluate via evaluate()
                IRGraph mini;
                NodeId m0 = mini.add_node(IRNode(0, OpKind::CONST, "c0", {}, c0.output_meta));
                mini.get_node_mut(m0).const_data = c0.const_data;
                NodeId m1 = mini.add_node(IRNode(1, OpKind::CONST, "c1", {}, c1.output_meta));
                mini.get_node_mut(m1).const_data = c1.const_data;
                NodeId m_op = mini.add_node(IRNode(2, OpKind::MATMUL, "mm", {m0, m1}, node.output_meta));
                mini.set_inputs({});
                mini.set_outputs({m_op});

                auto res = interp.evaluate(mini, {});
                folded_data = res[m_op];
            } else if (node.op == OpKind::ADD) {
                const auto& c0 = graph.get_node(node.inputs[0]);
                const auto& c1 = graph.get_node(node.inputs[1]);
                IRGraph mini;
                NodeId m0 = mini.add_node(IRNode(0, OpKind::CONST, "c0", {}, c0.output_meta));
                mini.get_node_mut(m0).const_data = c0.const_data;
                NodeId m1 = mini.add_node(IRNode(1, OpKind::CONST, "c1", {}, c1.output_meta));
                mini.get_node_mut(m1).const_data = c1.const_data;
                NodeId m_op = mini.add_node(IRNode(2, OpKind::ADD, "add", {m0, m1}, node.output_meta));
                mini.set_inputs({});
                mini.set_outputs({m_op});

                auto res = interp.evaluate(mini, {});
                folded_data = res[m_op];
            } else if (node.op == OpKind::RELU) {
                const auto& c0 = graph.get_node(node.inputs[0]);
                IRGraph mini;
                NodeId m0 = mini.add_node(IRNode(0, OpKind::CONST, "c0", {}, c0.output_meta));
                mini.get_node_mut(m0).const_data = c0.const_data;
                NodeId m_op = mini.add_node(IRNode(1, OpKind::RELU, "relu", {m0}, node.output_meta));
                mini.set_inputs({});
                mini.set_outputs({m_op});

                auto res = interp.evaluate(mini, {});
                folded_data = res[m_op];
            } else {
                continue;
            }

            // Create folded CONST node
            NodeId orig_id = node.id;
            std::string folded_name = "folded_" + node.name;
            TensorMeta folded_meta = node.output_meta;

            IRNode new_const;
            new_const.op = OpKind::CONST;
            new_const.name = std::move(folded_name);
            new_const.output_meta = std::move(folded_meta);
            new_const.const_data = std::move(folded_data);
            NodeId new_id = graph.add_node(std::move(new_const));

            graph.replace_all_uses(orig_id, new_id);
            graph.delete_node(orig_id);
            any_changed = true;
        }
    }
    return any_changed;
}

// ============================================================================
// Pass B: Algebraic Simplification
// ============================================================================
bool AlgebraicSimplificationPass::run(IRGraph& graph) {
    bool any_changed = false;

    auto order = graph.topological_order();
    for (NodeId id : order) {
        if (!graph.has_node(id)) continue;
        const auto& node = graph.get_node(id);
        if (node.deleted) continue;

        // Rule 1: ADD(X, 0) -> X when 0 is all zeros with matching shape.
        // Fast-math / signed-zero contract: Under IEEE 754, (-0.0f) + (+0.0f) = +0.0f.
        // ForgeTune ML inference adopts fast-math semantics (-fno-signed-zeros), treating +0.0f and -0.0f
        // as numerically equivalent.
        if (node.op == OpKind::ADD && node.inputs.size() == 2) {
            NodeId left_id = node.inputs[0];
            NodeId right_id = node.inputs[1];
            const auto& left = graph.get_node(left_id);
            const auto& right = graph.get_node(right_id);

            // Check if right is CONST zero with compatible shape
            if (right.op == OpKind::CONST && node.output_meta.shape == left.output_meta.shape) {
                bool all_zeros = true;
                for (float f : right.const_data) {
                    if (f != 0.0f) {
                        all_zeros = false;
                        break;
                    }
                }
                if (all_zeros) {
                    graph.replace_all_uses(node.id, left_id);
                    graph.delete_node(node.id);
                    any_changed = true;
                    continue;
                }
            }

            // Check if left is CONST zero with compatible shape
            if (left.op == OpKind::CONST && node.output_meta.shape == right.output_meta.shape) {
                bool all_zeros = true;
                for (float f : left.const_data) {
                    if (f != 0.0f) {
                        all_zeros = false;
                        break;
                    }
                }
                if (all_zeros) {
                    graph.replace_all_uses(node.id, right_id);
                    graph.delete_node(node.id);
                    any_changed = true;
                    continue;
                }
            }
        }

        // Rule 2: RELU(RELU(X)) -> RELU(X) idempotent property
        if (node.op == OpKind::RELU && node.inputs.size() == 1) {
            NodeId in_id = node.inputs[0];
            const auto& in_node = graph.get_node(in_id);
            if (in_node.op == OpKind::RELU) {
                // node is RELU(in_node). Since in_node is already a RELU, node is identical to in_node!
                graph.replace_all_uses(node.id, in_id);
                graph.delete_node(node.id);
                any_changed = true;
                continue;
            }
        }
    }
    return any_changed;
}

// ============================================================================
// Pass C: Dead-Code Elimination
// ============================================================================
bool DeadCodeEliminationPass::run(IRGraph& graph) {
    bool any_changed = false;

    // 1. Mark live nodes via backwards reachability from graph outputs
    std::unordered_set<NodeId> live;
    std::vector<NodeId> worklist;

    for (NodeId out_id : graph.get_outputs()) {
        if (graph.has_node(out_id) && !graph.get_node(out_id).deleted) {
            live.insert(out_id);
            worklist.push_back(out_id);
        }
    }
    // Preserve declared graph inputs as live roots to maintain entry parameter contract and verifier validity
    for (NodeId in_id : graph.get_inputs()) {
        if (graph.has_node(in_id) && !graph.get_node(in_id).deleted) {
            live.insert(in_id);
        }
    }
    while (!worklist.empty()) {
        NodeId curr = worklist.back();
        worklist.pop_back();

        const auto& node = graph.get_node(curr);
        for (NodeId inp : node.inputs) {
            if (graph.has_node(inp) && !graph.get_node(inp).deleted) {
                if (live.insert(inp).second) {
                    worklist.push_back(inp);
                }
            }
        }
    }

    // 2. Any non-live live node is dead -> delete it!
    for (const auto& node : graph.nodes()) {
        if (!node.deleted && live.find(node.id) == live.end()) {
            graph.delete_node(node.id);
            any_changed = true;
        }
    }

    return any_changed;
}

} // namespace forgetune
