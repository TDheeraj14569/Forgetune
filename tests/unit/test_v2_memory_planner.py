"""
Unit tests for ForgeTune v2.0 MemoryPlanner and Intermediate Buffer Reuse.
"""

from forgetune.graph_adapter import Node, Graph
from forgetune import _forgetune_cpp as cpp

def test_memory_planner_pipeline_reuse():
    # 3-layer sequential MLP:
    # x -> Linear1 -> ReLU1 -> Linear2 -> ReLU2 -> Output
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 32]))

    w1 = g.add_node(Node(1, "CONST", shape=[32, 32], const_data=[0.01] * 1024))
    b1 = g.add_node(Node(2, "CONST", shape=[32], const_data=[0.0] * 32))
    mm1 = g.add_node(Node(3, "MATMUL", inputs=[x, w1], shape=[-1, 32]))
    add1 = g.add_node(Node(4, "ADD", inputs=[mm1, b1], shape=[-1, 32]))
    relu1 = g.add_node(Node(5, "RELU", inputs=[add1], shape=[-1, 32]))

    w2 = g.add_node(Node(6, "CONST", shape=[32, 32], const_data=[0.01] * 1024))
    b2 = g.add_node(Node(7, "CONST", shape=[32], const_data=[0.0] * 32))
    mm2 = g.add_node(Node(8, "MATMUL", inputs=[relu1, w2], shape=[-1, 32]))
    add2 = g.add_node(Node(9, "ADD", inputs=[mm2, b2], shape=[-1, 32]))
    relu2 = g.add_node(Node(10, "RELU", inputs=[add2], shape=[-1, 32]))

    g.set_inputs([x])
    g.set_outputs([relu2])

    cpp_graph = cpp.json_to_graph(g.to_json())
    planner = cpp.MemoryPlanner(alignment=64)
    plan = planner.plan(cpp_graph, max_batch=128)

    # 1. Savings must be strictly positive on sequential pipeline
    assert plan.savings_percent > 0.0, f"Expected memory savings, got {plan.savings_percent}%"
    assert plan.arena_size_bytes < plan.naive_size_bytes

    # 2. All allocations must be 64-byte aligned
    for nid, alloc in plan.allocations.items():
        assert alloc.offset % 64 == 0, f"Allocation for node {nid} not 64-byte aligned: {alloc.offset}"
        assert alloc.size_bytes % 64 == 0, f"Size for node {nid} not 64-byte aligned: {alloc.size_bytes}"

    # 3. For any two allocations with overlapping lifespans, their memory intervals MUST NOT collide
    alloc_list = list(plan.allocations.values())
    for i in range(len(alloc_list)):
        for j in range(i + 1, len(alloc_list)):
            a1 = alloc_list[i]
            a2 = alloc_list[j]

            lifespans_overlap = not (a1.death_step < a2.birth_step or a2.death_step < a1.birth_step)
            if lifespans_overlap:
                start1 = a1.offset
                end1 = a1.offset + a1.size_bytes
                start2 = a2.offset
                end2 = a2.offset + a2.size_bytes

                # Verify spatial disjointness: [start1, end1) does not overlap [start2, end2)
                overlap = max(start1, start2) < min(end1, end2)
                assert not overlap, (
                    f"Memory collision between node {a1.node_id} (offset {start1}..{end1}, span [{a1.birth_step}, {a1.death_step}]) "
                    f"and node {a2.node_id} (offset {start2}..{end2}, span [{a2.birth_step}, {a2.death_step}])"
                )

def test_memory_planner_disjoint_buffer_reuse():
    # Construct graph with two clearly separated stages
    g = Graph()
    x = g.add_node(Node(0, "INPUT", shape=[-1, 64]))
    w1 = g.add_node(Node(1, "CONST", shape=[64, 64], const_data=[0.01] * 4096))
    mm1 = g.add_node(Node(2, "MATMUL", inputs=[x, w1], shape=[-1, 64]))
    r1 = g.add_node(Node(3, "RELU", inputs=[mm1], shape=[-1, 64]))

    w2 = g.add_node(Node(4, "CONST", shape=[64, 64], const_data=[0.01] * 4096))
    mm2 = g.add_node(Node(5, "MATMUL", inputs=[r1, w2], shape=[-1, 64]))
    r2 = g.add_node(Node(6, "RELU", inputs=[mm2], shape=[-1, 64]))
    g.set_inputs([x])
    g.set_outputs([r2])

    cpp_graph = cpp.json_to_graph(g.to_json())
    planner = cpp.MemoryPlanner(alignment=64)
    plan = planner.plan(cpp_graph, max_batch=128)

    # Node 2 (mm1) is consumed only by r1 (node 3). After r1 completes, mm1's memory can be reused by mm2 (node 5).
    alloc_mm1 = plan.allocations[2]
    alloc_mm2 = plan.allocations[5]

    assert alloc_mm1.death_step < alloc_mm2.birth_step
    # Both mm1 and mm2 should share offset 0!
    assert alloc_mm1.offset == alloc_mm2.offset == 0
