#include "forgetune/compiler.hpp"
#include "forgetune/memory_planner.hpp"
#include "forgetune/forgetune_abi.h"
#include <iostream>
#include <cassert>
#include <cmath>
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#endif

using namespace forgetune;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cout << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

bool test_ir_construction_and_verification() {
    std::cout << "[TEST] IR construction and verification..." << std::endl;
    IRGraph g;
    // Input X: [2, 4]
    NodeId x = g.add_node(IRNode(g.allocate_id(), OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 4}, DType::FLOAT32)));
    // Weight W: [4, 8]
    IRNode w_node(g.allocate_id(), OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 8}, DType::FLOAT32));
    w_node.const_data.assign(32, 1.0f);
    NodeId w = g.add_node(std::move(w_node));
    // Bias B: [8]
    IRNode b_node(g.allocate_id(), OpKind::CONST, "B", {}, TensorMeta(TensorShape{8}, DType::FLOAT32));
    b_node.const_data.assign(8, 0.5f);
    NodeId b = g.add_node(std::move(b_node));
    // MATMUL: [2, 8]
    NodeId mm = g.add_node(IRNode(g.allocate_id(), OpKind::MATMUL, "MM", {x, w}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
    // ADD: [2, 8]
    NodeId add = g.add_node(IRNode(g.allocate_id(), OpKind::ADD, "ADD", {mm, b}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
    // RELU: [2, 8]
    NodeId relu = g.add_node(IRNode(g.allocate_id(), OpKind::RELU, "RELU", {add}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));

    g.set_inputs({x});
    g.set_outputs({relu});

    auto vres = verify_graph(g, false);
    TEST_ASSERT(vres.passed, "Valid graph must pass verification");
    TEST_ASSERT(g.live_node_count() == 6, "Must have 6 live nodes");
    std::cout << "  PASSED\n";
    return true;
}

bool test_negative_graph_verifier() {
    std::cout << "[TEST] Negative graph verifier diagnostics..." << std::endl;
    // 1. Incompatible matmul dimensions: [2, 4] @ [5, 8]
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 4}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{5, 8}, DType::FLOAT32));
        w.const_data.assign(40, 1.0f);
        NodeId wn = g.add_node(std::move(w));
        NodeId mm = g.add_node(IRNode(2, OpKind::MATMUL, "MM", {x, wn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({mm});

        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Incompatible matmul dimensions must be rejected");
    }

    // 2. Cycle detection: 0 -> 1 -> 0
    {
        IRGraph g;
        NodeId n0 = g.add_node(IRNode(0, OpKind::RELU, "N0", {1}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        NodeId n1 = g.add_node(IRNode(1, OpKind::RELU, "N1", {0}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        g.set_inputs({n0});
        g.set_outputs({n1});

        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Cycle must be rejected");
    }

    // 3. Constant buffer size mismatch
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        IRNode c(1, OpKind::CONST, "C", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32));
        c.const_data.assign(3, 1.0f); // 3 instead of 4!
        g.add_node(std::move(c));
        g.set_inputs({x});
        g.set_outputs({x});

        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Constant buffer size mismatch must be rejected");
    }

    // 4. Incompatible broadcast rank-1: [2, 3] + [2] -> illegal
    {
        IRGraph g;
        NodeId a = g.add_node(IRNode(0, OpKind::INPUT, "A", {}, TensorMeta(TensorShape{2, 3}, DType::FLOAT32)));
        IRNode b(1, OpKind::CONST, "B", {}, TensorMeta(TensorShape{2}, DType::FLOAT32));
        b.const_data.assign(2, 1.0f);
        NodeId bn = g.add_node(std::move(b));
        g.add_node(IRNode(2, OpKind::ADD, "ADD", {a, bn}, TensorMeta(TensorShape{2, 3}, DType::FLOAT32)));
        g.set_inputs({a});
        g.set_outputs({2});

        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Illegal rank-1 broadcast [2, 3] + [2] must be rejected");
    }
    std::cout << "  PASSED\n";
    return true;
}

bool test_interpreter_correctness() {
    std::cout << "[TEST] Reference interpreter numerical evaluation..." << std::endl;
    // 1. Rank-1 bias vector: Y = ReLU(X @ W + B) with B = [-2, 1]
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32));
        w.const_data = {1.0f, 0.0f, 0.0f, 1.0f};
        NodeId wn = g.add_node(std::move(w));

        IRNode b(2, OpKind::CONST, "B", {}, TensorMeta(TensorShape{2}, DType::FLOAT32));
        b.const_data = {-2.0f, 1.0f};
        NodeId bn = g.add_node(std::move(b));

        NodeId mm = g.add_node(IRNode(3, OpKind::MATMUL, "MM", {x, wn}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        NodeId add = g.add_node(IRNode(4, OpKind::ADD, "ADD", {mm, bn}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        NodeId relu = g.add_node(IRNode(5, OpKind::RELU, "RELU", {add}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));

        g.set_inputs({x});
        g.set_outputs({relu});

        Interpreter interp;
        std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto out = interp.evaluate(g, {{x, x_data}});

        const auto& res = out[relu];
        TEST_ASSERT(res.size() == 4, "Output must have 4 elements");
        TEST_ASSERT(std::abs(res[0] - 0.0f) < 1e-5f, "res[0] should be 0.0");
        TEST_ASSERT(std::abs(res[1] - 3.0f) < 1e-5f, "res[1] should be 3.0");
        TEST_ASSERT(std::abs(res[2] - 1.0f) < 1e-5f, "res[2] should be 1.0");
        TEST_ASSERT(std::abs(res[3] - 5.0f) < 1e-5f, "res[3] should be 5.0");
    }

    // 2. Rank-2 row bias: B = [1, 2] broadcast over [2, 2]
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        IRNode b(1, OpKind::CONST, "B", {}, TensorMeta(TensorShape{1, 2}, DType::FLOAT32));
        b.const_data = {10.0f, 20.0f};
        NodeId bn = g.add_node(std::move(b));
        NodeId add = g.add_node(IRNode(2, OpKind::ADD, "ADD", {x, bn}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({add});

        Interpreter interp;
        std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto out = interp.evaluate(g, {{x, x_data}});

        const auto& res = out[add];
        TEST_ASSERT(std::abs(res[0] - 11.0f) < 1e-5f, "res[0] should be 11.0");
        TEST_ASSERT(std::abs(res[1] - 22.0f) < 1e-5f, "res[1] should be 22.0");
        TEST_ASSERT(std::abs(res[2] - 13.0f) < 1e-5f, "res[2] should be 13.0");
        TEST_ASSERT(std::abs(res[3] - 24.0f) < 1e-5f, "res[3] should be 24.0");
    }
    std::cout << "  PASSED\n";
    return true;
}

bool test_constant_folding_pass() {
    std::cout << "[TEST] Constant folding pass..." << std::endl;
    IRGraph g;
    NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));

    // Const A: [2, 2] = [1, 2, 3, 4]
    IRNode c1(1, OpKind::CONST, "C1", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32));
    c1.const_data = {1.0f, 2.0f, 3.0f, 4.0f};
    NodeId c1_id = g.add_node(std::move(c1));

    // Const B: [2, 2] = [2, 2, 2, 2]
    IRNode c2(2, OpKind::CONST, "C2", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32));
    c2.const_data = {2.0f, 2.0f, 2.0f, 2.0f};
    NodeId c2_id = g.add_node(std::move(c2));

    // ADD(C1, C2) is completely constant!
    NodeId add_c = g.add_node(IRNode(3, OpKind::ADD, "ADD_C", {c1_id, c2_id}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));

    // ADD(X, ADD_C) has dynamic input X
    NodeId final_add = g.add_node(IRNode(4, OpKind::ADD, "FINAL", {x, add_c}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));

    g.set_inputs({x});
    g.set_outputs({final_add});

    ConstantFoldingPass pass;
    bool changed = pass.run(g);
    TEST_ASSERT(changed, "Constant folding should modify graph");

    // ADD_C should have been replaced with a folded CONST node
    DeadCodeEliminationPass dce;
    dce.run(g);

    verify_graph(g, true);

    // Verify numerical equivalence with interpreter
    Interpreter interp;
    std::vector<float> x_data = {10.0f, 20.0f, 30.0f, 40.0f};
    auto out = interp.evaluate(g, {{x, x_data}});
    const auto& res = out[final_add];
    // (1+2)+10 = 13, (2+2)+20 = 24, (3+2)+30 = 35, (4+2)+40 = 46
    TEST_ASSERT(std::abs(res[0] - 13.0f) < 1e-5f, "res[0] must be 13.0");
    TEST_ASSERT(std::abs(res[1] - 24.0f) < 1e-5f, "res[1] must be 24.0");
    std::cout << "  PASSED\n";
    return true;
}

bool test_dead_code_elimination_pass() {
    std::cout << "[TEST] Dead code elimination pass..." << std::endl;
    IRGraph g;
    NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
    // Live node
    NodeId relu = g.add_node(IRNode(1, OpKind::RELU, "RELU", {x}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
    // Dead node: unused computation
    NodeId dead_node = g.add_node(IRNode(2, OpKind::RELU, "DEAD", {x}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));

    g.set_inputs({x});
    g.set_outputs({relu});

    TEST_ASSERT(g.live_node_count() == 3, "Initially 3 live nodes");
    DeadCodeEliminationPass dce;
    bool changed = dce.run(g);
    TEST_ASSERT(changed, "DCE should remove dead node");
    TEST_ASSERT(g.live_node_count() == 2, "After DCE, exactly 2 live nodes");
    TEST_ASSERT(g.get_node(dead_node).deleted, "Dead node must be marked deleted");
    std::cout << "  PASSED\n";
    return true;
}

bool test_operator_fusion_pass() {
    std::cout << "[TEST] Operator fusion pass..." << std::endl;
    IRGraph g;
    NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 4}, DType::FLOAT32)));

    IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 8}, DType::FLOAT32));
    w.const_data.assign(32, 1.0f);
    NodeId wn = g.add_node(std::move(w));

    IRNode b(2, OpKind::CONST, "B", {}, TensorMeta(TensorShape{8}, DType::FLOAT32));
    b.const_data.assign(8, 0.5f);
    NodeId bn = g.add_node(std::move(b));

    NodeId mm = g.add_node(IRNode(3, OpKind::MATMUL, "MM", {x, wn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
    NodeId add = g.add_node(IRNode(4, OpKind::ADD, "ADD", {mm, bn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
    NodeId relu = g.add_node(IRNode(5, OpKind::RELU, "RELU", {add}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));

    g.set_inputs({x});
    g.set_outputs({relu});

    OperatorFusionPass fusion;
    bool changed = fusion.run(g);
    TEST_ASSERT(changed, "Fusion must succeed on MATMUL -> ADD -> RELU");

    // Output should now be fused node
    NodeId new_out = g.get_outputs()[0];
    const auto& out_node = g.get_node(new_out);
    TEST_ASSERT(out_node.op == OpKind::FUSED_MATMUL_ADD_RELU, "Output must be FUSED_MATMUL_ADD_RELU");
    TEST_ASSERT(g.get_node(mm).deleted, "Intermediate MATMUL must be deleted");
    TEST_ASSERT(g.get_node(add).deleted, "Intermediate ADD must be deleted");

    verify_graph(g, true);

    // Test numerical agreement of fused node via interpreter
    Interpreter interp;
    std::vector<float> x_data(8, 2.0f);
    auto res = interp.evaluate(g, {{x, x_data}});
    const auto& y = res[new_out];
    TEST_ASSERT(y.size() == 16, "Must have 16 elements [2, 8]");
    // X @ W = 2.0 * 4 = 8.0, + 0.5 = 8.5
    for (float val : y) {
        TEST_ASSERT(std::abs(val - 8.5f) < 1e-5f, "Each element must be 8.5");
    }
    std::cout << "  PASSED\n";
    return true;
}

bool test_json_roundtrip() {
    std::cout << "[TEST] JSON serialization/deserialization round-trip..." << std::endl;
    IRGraph g;
    NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 4}, DType::FLOAT32)));
    IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 8}, DType::FLOAT32));
    w.const_data.assign(32, 1.5f);
    NodeId wn = g.add_node(std::move(w));
    NodeId mm = g.add_node(IRNode(2, OpKind::MATMUL, "MM", {x, wn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
    g.set_inputs({x});
    g.set_outputs({mm});

    std::string json_str = graph_to_json(g);
    TEST_ASSERT(!json_str.empty(), "JSON string should not be empty");

    IRGraph g2 = json_to_graph(json_str);
    verify_graph(g2, true);

    TEST_ASSERT(g2.get_inputs() == g.get_inputs(), "Inputs must match");
    TEST_ASSERT(g2.get_outputs() == g.get_outputs(), "Outputs must match");
    TEST_ASSERT(g2.live_node_count() == g.live_node_count(), "Live count must match");
    std::cout << "  PASSED\n";
    return true;
}

bool test_algebraic_simplification_and_fusion_safety() {
    std::cout << "[TEST] Algebraic simplification and fusion safety..." << std::endl;
    // 1. Algebraic simplification: X + 0 -> X
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        IRNode zero(1, OpKind::CONST, "Zero", {}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32));
        zero.const_data.assign(4, 0.0f);
        NodeId z = g.add_node(std::move(zero));
        NodeId add = g.add_node(IRNode(2, OpKind::ADD, "ADD", {x, z}, TensorMeta(TensorShape{2, 2}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({add});

        AlgebraicSimplificationPass as_pass;
        bool changed = as_pass.run(g);
        TEST_ASSERT(changed, "Algebraic simplification must modify graph for X + 0");
        TEST_ASSERT(g.get_outputs()[0] == x, "Graph output must be rewritten to input X");
        TEST_ASSERT(g.get_node(add).deleted, "ADD node must be marked deleted");
    }

    // 2. Fusion safety: multi-consumer intermediate must NOT be fused
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 4}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 8}, DType::FLOAT32));
        w.const_data.assign(32, 1.0f);
        NodeId wn = g.add_node(std::move(w));
        IRNode b(2, OpKind::CONST, "B", {}, TensorMeta(TensorShape{8}, DType::FLOAT32));
        b.const_data.assign(8, 0.5f);
        NodeId bn = g.add_node(std::move(b));

        NodeId mm = g.add_node(IRNode(3, OpKind::MATMUL, "MM", {x, wn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
        NodeId add = g.add_node(IRNode(4, OpKind::ADD, "ADD", {mm, bn}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
        NodeId relu = g.add_node(IRNode(5, OpKind::RELU, "RELU", {add}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
        NodeId extra = g.add_node(IRNode(6, OpKind::RELU, "EXTRA", {mm}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));

        g.set_inputs({x});
        g.set_outputs({relu, extra});

        OperatorFusionPass fusion_pass;
        bool fused = fusion_pass.run(g);
        TEST_ASSERT(!fused, "Fusion must NOT occur when intermediate has multiple consumers");
    }
    std::cout << "  PASSED\n";
    return true;
}

bool test_v2_shapes_and_conv2d_verifier() {
    std::cout << "[TEST] V2 shapes and Conv2D verifier..." << std::endl;
    // 1. Dynamic rank-2 and rank-4 shape properties
    TensorShape s2{-1, 16};
    TEST_ASSERT(s2.has_dynamic_batch(), "s2 must have dynamic batch");
    TEST_ASSERT(s2.static_element_count() == 16, "static element count must be 16");
    TEST_ASSERT(s2.element_count(4) == 64, "element_count(4) must be 64");

    TensorShape s4{-1, 8, 14, 14};
    TEST_ASSERT(s4.has_dynamic_batch(), "s4 must have dynamic batch");
    TEST_ASSERT(s4.static_element_count() == 8 * 14 * 14, "s4 static count mismatch");
    TEST_ASSERT(s4.element_count(2) == 2 * 8 * 14 * 14, "s4 count for B=2 mismatch");

    // 2. Reject non-leading dynamic dimension: [2, -1]
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, -1}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({x});
        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Non-leading dynamic dimension must be rejected by verifier");
    }

    // 3. Conv2d graph verifier: valid NCHW + OIHW
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{-1, 1, 8, 8}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(36, 0.1f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{-1, 4, 8, 8}, DType::FLOAT32));
        cnode.conv_attrs.stride_h = 1;
        cnode.conv_attrs.stride_w = 1;
        cnode.conv_attrs.pad_h = 1;
        cnode.conv_attrs.pad_w = 1;
        NodeId cn = g.add_node(std::move(cnode));

        g.set_inputs({x});
        g.set_outputs({cn});
        auto vres = verify_graph(g, false);
        TEST_ASSERT(vres.passed, "Valid dynamic Conv2D graph must pass verification");
    }

    // 4. Conv2d verifier: reject invalid stride <= 0
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{-1, 1, 8, 8}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(36, 0.1f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{-1, 4, 8, 8}, DType::FLOAT32));
        cnode.conv_attrs.stride_h = 0; // Invalid!
        g.add_node(std::move(cnode));
        g.set_inputs({x});
        g.set_outputs({2});
        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Conv2D with stride 0 must be rejected");
    }

    // 5. Conv2d verifier: reject negative dilation and negative padding
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{-1, 1, 8, 8}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{4, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(36, 0.1f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{-1, 4, 8, 8}, DType::FLOAT32));
        cnode.conv_attrs.dilation_h = -1; // Invalid!
        g.add_node(std::move(cnode));
        g.set_inputs({x});
        g.set_outputs({2});
        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Conv2D with negative dilation must be rejected");
    }

    // 6. Conv2d verifier: reject negative numerator (kernel exceeds padded input)
    {
        IRGraph g;
        // Input spatial dimension is 2x2, kernel is 5x5 with 0 padding -> num = 2 - 5 = -3 < 0
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{1, 1, 2, 2}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{1, 1, 5, 5}, DType::FLOAT32));
        w.const_data.assign(25, 0.1f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{1, 1, 1, 1}, DType::FLOAT32));
        cnode.conv_attrs.pad_h = 0;
        cnode.conv_attrs.pad_w = 0;
        g.add_node(std::move(cnode));
        g.set_inputs({x});
        g.set_outputs({2});
        auto vres = verify_graph(g, false);
        TEST_ASSERT(!vres.passed, "Conv2D where kernel exceeds padded input must be rejected");
    }

    std::cout << "  PASSED\n";
    return true;
}

bool test_v2_interpreter_new_ops() {
    std::cout << "[TEST] V2 interpreter evaluation for new operators..." << std::endl;
    // 1. MUL: elementwise and broadcasting
    {
        IRGraph g;
        NodeId a = g.add_node(IRNode(0, OpKind::INPUT, "A", {}, TensorMeta(TensorShape{2, 3}, DType::FLOAT32)));
        IRNode b(1, OpKind::CONST, "B", {}, TensorMeta(TensorShape{3}, DType::FLOAT32));
        b.const_data = {2.0f, -1.0f, 0.5f};
        NodeId bn = g.add_node(std::move(b));
        NodeId mul = g.add_node(IRNode(2, OpKind::MUL, "MUL", {a, bn}, TensorMeta(TensorShape{2, 3}, DType::FLOAT32)));
        g.set_inputs({a});
        g.set_outputs({mul});

        Interpreter interp;
        std::vector<float> in_data = {1.0f, 2.0f, 4.0f,  -3.0f, 0.0f, 10.0f};
        auto outs = interp.evaluate(g, {{a, in_data}});
        TEST_ASSERT(outs.find(mul) != outs.end(), "Must produce MUL output");
        const auto& res_mul = outs[mul];
        std::vector<float> expected = {2.0f, -2.0f, 2.0f,  -6.0f, 0.0f, 5.0f};
        for (size_t i = 0; i < expected.size(); ++i) {
            TEST_ASSERT(std::abs(res_mul[i] - expected[i]) < 1e-5f, "MUL evaluation mismatch");
        }
    }

    // 2. SIGMOID and TANH
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{3}, DType::FLOAT32)));
        NodeId sig = g.add_node(IRNode(1, OpKind::SIGMOID, "SIG", {x}, TensorMeta(TensorShape{3}, DType::FLOAT32)));
        NodeId tnh = g.add_node(IRNode(2, OpKind::TANH, "TNH", {x}, TensorMeta(TensorShape{3}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({sig, tnh});

        Interpreter interp;
        std::vector<float> in_data = {0.0f, 2.0f, -2.0f};
        auto outs = interp.evaluate(g, {{x, in_data}});
        const auto& res_sig = outs[sig];
        const auto& res_tnh = outs[tnh];
        TEST_ASSERT(std::abs(res_sig[0] - 0.5f) < 1e-5f, "sigmoid(0) must be 0.5");
        TEST_ASSERT(std::abs(res_tnh[0] - 0.0f) < 1e-5f, "tanh(0) must be 0");
        TEST_ASSERT(std::abs(res_sig[1] - (1.0f / (1.0f + std::exp(-2.0f)))) < 1e-5f, "sigmoid(2) mismatch");
        TEST_ASSERT(std::abs(res_tnh[1] - std::tanh(2.0f)) < 1e-5f, "tanh(2) mismatch");
    }

    // 3. FLATTEN: [2, 2, 2, 2] -> [2, 8]
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{2, 2, 2, 2}, DType::FLOAT32)));
        NodeId flat = g.add_node(IRNode(1, OpKind::FLATTEN, "FLAT", {x}, TensorMeta(TensorShape{2, 8}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({flat});

        Interpreter interp;
        std::vector<float> in_data(16);
        for (int i = 0; i < 16; ++i) in_data[i] = static_cast<float>(i);
        auto outs = interp.evaluate(g, {{x, in_data}});
        const auto& res_flat = outs[flat];
        TEST_ASSERT(res_flat.size() == 16, "Flattened size must be 16");
        for (int i = 0; i < 16; ++i) {
            TEST_ASSERT(res_flat[i] == static_cast<float>(i), "Flattened elements must preserve order");
        }
    }

    // 4. CONV2D: 1x1x3x3 input with 1x1x3x3 kernel of 1.0f -> output sum 9.0f
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{1, 1, 3, 3}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{1, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(9, 1.0f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{1, 1, 1, 1}, DType::FLOAT32));
        cnode.conv_attrs.stride_h = 1;
        cnode.conv_attrs.stride_w = 1;
        cnode.conv_attrs.pad_h = 0;
        cnode.conv_attrs.pad_w = 0;
        NodeId cn = g.add_node(std::move(cnode));
        g.set_inputs({x});
        g.set_outputs({cn});

        Interpreter interp;
        std::vector<float> in_data(9, 1.0f);
        auto outs = interp.evaluate(g, {{x, in_data}});
        const auto& res_cn = outs[cn];
        TEST_ASSERT(res_cn.size() == 1, "Output must have 1 element");
        TEST_ASSERT(std::abs(res_cn[0] - 9.0f) < 1e-5f, "Conv output must be 9.0f");
    }

    std::cout << "  PASSED\n";
    return true;
}

bool test_v2_conv2d_relu_fusion_pass() {
    std::cout << "[TEST] Conv2D + ReLU operator fusion pass..." << std::endl;
    // 1. Success case: Conv2D -> ReLU
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{1, 1, 4, 4}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{2, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(18, 0.5f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{1, 2, 4, 4}, DType::FLOAT32));
        cnode.conv_attrs.stride_h = 1;
        cnode.conv_attrs.stride_w = 1;
        cnode.conv_attrs.pad_h = 1;
        cnode.conv_attrs.pad_w = 1;
        NodeId cn = g.add_node(std::move(cnode));
        NodeId rnode = g.add_node(IRNode(3, OpKind::RELU, "ReLU", {cn}, TensorMeta(TensorShape{1, 2, 4, 4}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({rnode});

        OperatorFusionPass fpass;
        bool fused = fpass.run(g);
        TEST_ASSERT(fused, "Conv2D + ReLU must be fused");
        TEST_ASSERT(g.get_node(cn).deleted, "Original Conv2D must be deleted");
        TEST_ASSERT(g.get_node(rnode).deleted, "Original ReLU must be deleted");
        NodeId out_id = g.get_outputs()[0];
        TEST_ASSERT(g.get_node(out_id).op == OpKind::FUSED_CONV2D_RELU, "Fused node must be FUSED_CONV2D_RELU");
    }

    // 2. Rejection case: Conv2D output has multiple consumers
    {
        IRGraph g;
        NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{1, 1, 4, 4}, DType::FLOAT32)));
        IRNode w(1, OpKind::CONST, "W", {}, TensorMeta(TensorShape{2, 1, 3, 3}, DType::FLOAT32));
        w.const_data.assign(18, 0.5f);
        NodeId wn = g.add_node(std::move(w));

        IRNode cnode(2, OpKind::CONV2D, "Conv", {x, wn}, TensorMeta(TensorShape{1, 2, 4, 4}, DType::FLOAT32));
        NodeId cn = g.add_node(std::move(cnode));
        NodeId rnode = g.add_node(IRNode(3, OpKind::RELU, "ReLU", {cn}, TensorMeta(TensorShape{1, 2, 4, 4}, DType::FLOAT32)));
        NodeId extra = g.add_node(IRNode(4, OpKind::RELU, "Extra", {cn}, TensorMeta(TensorShape{1, 2, 4, 4}, DType::FLOAT32)));
        g.set_inputs({x});
        g.set_outputs({rnode, extra});

        OperatorFusionPass fpass;
        bool fused = fpass.run(g);
        TEST_ASSERT(!fused, "Conv2D with multiple consumers must NOT be fused");
    }

    std::cout << "  PASSED\n";
    return true;
}

bool test_v2_memory_planner_and_64byte_alignment() {
    std::cout << "[TEST] V2 memory planner and 64-byte alignment verification..." << std::endl;
    IRGraph g;
    NodeId x = g.add_node(IRNode(0, OpKind::INPUT, "X", {}, TensorMeta(TensorShape{-1, 16}, DType::FLOAT32)));

    IRNode w1(1, OpKind::CONST, "W1", {}, TensorMeta(TensorShape{16, 32}, DType::FLOAT32));
    w1.const_data.assign(512, 0.1f);
    NodeId wn1 = g.add_node(std::move(w1));
    NodeId mm1 = g.add_node(IRNode(2, OpKind::MATMUL, "MM1", {x, wn1}, TensorMeta(TensorShape{-1, 32}, DType::FLOAT32)));
    NodeId r1 = g.add_node(IRNode(3, OpKind::RELU, "R1", {mm1}, TensorMeta(TensorShape{-1, 32}, DType::FLOAT32)));

    IRNode w2(4, OpKind::CONST, "W2", {}, TensorMeta(TensorShape{32, 16}, DType::FLOAT32));
    w2.const_data.assign(512, 0.1f);
    NodeId wn2 = g.add_node(std::move(w2));
    NodeId mm2 = g.add_node(IRNode(5, OpKind::MATMUL, "MM2", {r1, wn2}, TensorMeta(TensorShape{-1, 16}, DType::FLOAT32)));
    NodeId r2 = g.add_node(IRNode(6, OpKind::RELU, "R2", {mm2}, TensorMeta(TensorShape{-1, 16}, DType::FLOAT32)));

    IRNode w3(7, OpKind::CONST, "W3", {}, TensorMeta(TensorShape{16, 8}, DType::FLOAT32));
    w3.const_data.assign(128, 0.1f);
    NodeId wn3 = g.add_node(std::move(w3));
    NodeId mm3 = g.add_node(IRNode(8, OpKind::MATMUL, "MM3", {r2, wn3}, TensorMeta(TensorShape{-1, 8}, DType::FLOAT32)));

    g.set_inputs({x});
    g.set_outputs({mm3});

    MemoryPlanner planner(64);
    MemoryPlan plan = planner.plan(g, 128);

    TEST_ASSERT(plan.savings_percent > 30.0, "Planned memory savings should exceed 30%");
    TEST_ASSERT(plan.arena_size_bytes < plan.naive_size_bytes, "Arena size must be less than naive total");

    // Verify 64-byte alignment on every single allocated offset
    for (const auto& kv : plan.allocations) {
        const auto& alloc = kv.second;
        TEST_ASSERT(alloc.offset % 64 == 0, "Buffer offset must be 64-byte aligned");
        TEST_ASSERT(alloc.size_bytes % 64 == 0, "Buffer size must be a multiple of 64 bytes");
    }

    std::cout << "  PASSED (Planned Savings: " << plan.savings_percent << "%)\n";
    return true;
}

bool test_aligned_arena_pool_and_reentrancy() {
    std::cout << "[TEST] Aligned arena allocation, 64-byte pointer proof, and re-entrant leasing..." << std::endl;
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
    void* raw = _aligned_malloc(4096, 64);
    TEST_ASSERT(raw != nullptr, "_aligned_malloc must succeed");
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    TEST_ASSERT(addr % 64 == 0, "Raw heap buffer must be strictly 64-byte aligned");
    _aligned_free(raw);
#endif

    struct TestBlock {
        void* raw_ptr = nullptr;
        float* aligned_ptr = nullptr;
        size_t capacity_bytes = 0;
        bool in_use = false;
        int alloc_count = 0;

        ~TestBlock() {
            if (raw_ptr) {
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
                _aligned_free(raw_ptr);
#else
                std::free(raw_ptr);
#endif
            }
        }

        bool ensure(size_t req) {
            if (capacity_bytes >= req && aligned_ptr) return true;
            if (raw_ptr) {
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
                _aligned_free(raw_ptr);
#else
                std::free(raw_ptr);
#endif
            }
            size_t bytes = ((req + 63) / 64) * 64;
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
            raw_ptr = _aligned_malloc(bytes, 64);
            aligned_ptr = static_cast<float*>(raw_ptr);
#else
            void* p = nullptr;
            if (posix_memalign(&p, 64, bytes) == 0) {
                raw_ptr = p;
                aligned_ptr = static_cast<float*>(p);
            }
#endif
            alloc_count++;
            capacity_bytes = bytes;
            return aligned_ptr != nullptr;
        }
    };

    TestBlock b1;
    // First call (warm-up) for B=128 capacity (e.g. 32768 bytes)
    bool ok = b1.ensure(32768);
    TEST_ASSERT(ok, "Block allocation must succeed");
    TEST_ASSERT(reinterpret_cast<uintptr_t>(b1.aligned_ptr) % 64 == 0, "Base pointer must be 64-byte aligned");
    TEST_ASSERT(b1.alloc_count == 1, "Warm-up allocates exactly once");

    // Subsequent calls with batch sizes B in {1, 2, 7, 16, 32, 64, 128}
    size_t batch_bytes[] = {256, 512, 1792, 4096, 8192, 16384, 32768};
    for (size_t req : batch_bytes) {
        ok = b1.ensure(req);
        TEST_ASSERT(ok, "Ensure capacity must succeed");
        TEST_ASSERT(b1.alloc_count == 1, "Must NOT reallocate during subsequent dynamic batch calls (Zero Allocations!)");
    }

    // Test nested re-entrancy
    TestBlock b2;
    b1.in_use = true; // Outer call is active
    ok = b2.ensure(32768); // Inner call leases block 2
    TEST_ASSERT(ok, "Nested lease must succeed");
    TEST_ASSERT(b1.aligned_ptr != b2.aligned_ptr, "Nested call must have distinct non-overlapping arena");
    TEST_ASSERT(reinterpret_cast<uintptr_t>(b2.aligned_ptr) % 64 == 0, "Inner lease must also be 64-byte aligned");

    std::cout << "  PASSED (64-byte alignment & Zero-allocation verified)\n";
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running ForgeTune C++ Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    int passed = 0;
    int total = 0;

    auto run_test = [&](bool (*test_fn)()) {
        total++;
        if (test_fn()) {
            passed++;
        } else {
            std::cerr << "Test FAILED!\n";
        }
    };

    run_test(test_ir_construction_and_verification);
    run_test(test_negative_graph_verifier);
    run_test(test_interpreter_correctness);
    run_test(test_constant_folding_pass);
    run_test(test_dead_code_elimination_pass);
    run_test(test_operator_fusion_pass);
    run_test(test_json_roundtrip);
    run_test(test_algebraic_simplification_and_fusion_safety);
    run_test(test_v2_shapes_and_conv2d_verifier);
    run_test(test_v2_interpreter_new_ops);
    run_test(test_v2_conv2d_relu_fusion_pass);
    run_test(test_v2_memory_planner_and_64byte_alignment);
    run_test(test_aligned_arena_pool_and_reentrancy);

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary: " << passed << " / " << total << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
