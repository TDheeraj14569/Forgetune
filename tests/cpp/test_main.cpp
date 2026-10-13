#include "forgetune/compiler.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace forgetune;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
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

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary: " << passed << " / " << total << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
