#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "forgetune/ir.hpp"
#include "forgetune/graph_verifier.hpp"
#include "forgetune/interpreter.hpp"
#include "forgetune/json_io.hpp"
#include "forgetune/compiler.hpp"

namespace py = pybind11;
using namespace forgetune;

PYBIND11_MODULE(_forgetune_cpp, m) {
    m.doc() = "ForgeTune native C++17 compiler engine and reference interpreter bindings";

    // Schema version
    m.attr("IR_SCHEMA_VERSION_MAJOR") = IR_SCHEMA_VERSION_MAJOR;
    m.attr("IR_SCHEMA_VERSION_MINOR") = IR_SCHEMA_VERSION_MINOR;

    // DType
    py::enum_<DType>(m, "DType")
        .value("FLOAT32", DType::FLOAT32)
        .export_values();

    // OpKind
    py::enum_<OpKind>(m, "OpKind")
        .value("INPUT", OpKind::INPUT)
        .value("CONST", OpKind::CONST)
        .value("MATMUL", OpKind::MATMUL)
        .value("ADD", OpKind::ADD)
        .value("RELU", OpKind::RELU)
        .value("OUTPUT", OpKind::OUTPUT)
        .value("FUSED_MATMUL_ADD_RELU", OpKind::FUSED_MATMUL_ADD_RELU)
        .export_values();

    // TensorShape
    py::class_<TensorShape>(m, "TensorShape")
        .def(py::init<>())
        .def(py::init<std::vector<int64_t>>())
        .def_readwrite("dims", &TensorShape::dims)
        .def("rank", &TensorShape::rank)
        .def("element_count", &TensorShape::element_count)
        .def("__repr__", &TensorShape::to_string);

    // TensorMeta
    py::class_<TensorMeta>(m, "TensorMeta")
        .def(py::init<>())
        .def(py::init<TensorShape, DType>())
        .def_readwrite("shape", &TensorMeta::shape)
        .def_readwrite("dtype", &TensorMeta::dtype);

    // IRNode
    py::class_<IRNode>(m, "IRNode")
        .def(py::init<>())
        .def(py::init<NodeId, OpKind, std::string, std::vector<NodeId>, TensorMeta>())
        .def_readwrite("id", &IRNode::id)
        .def_readwrite("op", &IRNode::op)
        .def_readwrite("name", &IRNode::name)
        .def_readwrite("inputs", &IRNode::inputs)
        .def_readwrite("output_meta", &IRNode::output_meta)
        .def_readwrite("const_data", &IRNode::const_data)
        .def_readwrite("deleted", &IRNode::deleted)
        .def("__repr__", &IRNode::to_string);

    // IRGraph
    py::class_<IRGraph>(m, "IRGraph")
        .def(py::init<>())
        .def("add_node", &IRGraph::add_node)
        .def("get_node", &IRGraph::get_node, py::return_value_policy::reference)
        .def("has_node", &IRGraph::has_node)
        .def("set_inputs", &IRGraph::set_inputs)
        .def("set_outputs", &IRGraph::set_outputs)
        .def("get_inputs", &IRGraph::get_inputs)
        .def("get_outputs", &IRGraph::get_outputs)
        .def("topological_order", &IRGraph::topological_order)
        .def("live_node_count", &IRGraph::live_node_count)
        .def("allocate_id", &IRGraph::allocate_id)
        .def("to_string", &IRGraph::to_string)
        .def("__repr__", &IRGraph::to_string);

    // Verification
    py::class_<VerificationResult>(m, "VerificationResult")
        .def_readonly("passed", &VerificationResult::passed)
        .def_readonly("errors", &VerificationResult::errors)
        .def("to_string", &VerificationResult::to_string);

    m.def("verify_graph", &verify_graph,
          py::arg("graph"), py::arg("check_throw") = true,
          "Verify graph structure, shapes, arity, and acyclicity");

    // Interpreter
    py::class_<Interpreter>(m, "Interpreter")
        .def(py::init<>())
        .def("evaluate", [](const Interpreter& interp, const IRGraph& graph,
                            const std::unordered_map<NodeId, py::array_t<float>>& user_inputs) {
            std::unordered_map<NodeId, std::vector<float>> input_vecs;
            for (const auto& kv : user_inputs) {
                auto buf = kv.second.request();
                const float* ptr = static_cast<const float*>(buf.ptr);
                input_vecs[kv.first] = std::vector<float>(ptr, ptr + buf.size);
            }

            auto out_vecs = interp.evaluate(graph, input_vecs);

            std::unordered_map<NodeId, py::array_t<float>> results;
            for (const auto& kv : out_vecs) {
                const auto& node = graph.get_node(kv.first);
                const auto& dims = node.output_meta.shape.dims;
                std::vector<ssize_t> shape(dims.begin(), dims.end());
                py::array_t<float> arr(shape);
                auto buf = arr.request();
                std::memcpy(buf.ptr, kv.second.data(), kv.second.size() * sizeof(float));
                results[kv.first] = arr;
            }
            return results;
        }, py::arg("graph"), py::arg("user_inputs"),
        "Evaluates the IR graph directly using the C++ reference interpreter");

    // JSON IO
    m.def("graph_to_json", &graph_to_json, "Serialize an IRGraph to JSON");
    m.def("json_to_graph", &json_to_graph, "Deserialize an IRGraph from JSON");
}
