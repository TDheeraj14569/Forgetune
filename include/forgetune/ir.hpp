#ifndef FORGETUNE_IR_HPP
#define FORGETUNE_IR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include <limits>
#include <cstddef>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <functional>

namespace forgetune {

// ============================================================================
// Schema version for the IR format (used in serialization and cache keys)
// ============================================================================
constexpr int IR_SCHEMA_VERSION_MAJOR = 2;
constexpr int IR_SCHEMA_VERSION_MINOR = 0;

// ============================================================================
// Supported data types
// ============================================================================
enum class DType {
    FLOAT32,
    // Future: FLOAT64, INT32, etc.
};

inline const char* dtype_to_string(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return "f32";
        default: return "unknown";
    }
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::FLOAT32: return 4;
        default: throw std::runtime_error("Unknown dtype");
    }
}

// ============================================================================
// Tensor shape representation (Static and Symbolic Dynamic Batch)
// ============================================================================
struct TensorShape {
    static constexpr int64_t DYNAMIC_DIM = -1;
    static constexpr int64_t DEFAULT_MAX_BATCH = 128;

    std::vector<int64_t> dims;

    TensorShape() = default;
    explicit TensorShape(std::vector<int64_t> d) : dims(std::move(d)) {}
    TensorShape(std::initializer_list<int64_t> d) : dims(d) {}

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }

    bool is_dynamic() const {
        for (auto d : dims) {
            if (d == DYNAMIC_DIM) return true;
        }
        return false;
    }

    bool has_dynamic_batch() const {
        return (!dims.empty() && dims[0] == DYNAMIC_DIM);
    }

    int64_t static_element_count() const {
        if (dims.empty()) return 0;
        int64_t count = 1;
        size_t start = (is_dynamic() ? 1 : 0);
        for (size_t i = start; i < dims.size(); ++i) {
            int64_t d = dims[i];
            if (d <= 0) {
                throw std::runtime_error("TensorShape: non-positive dimension encountered");
            }
            if (count > (INT64_MAX / d)) {
                throw std::runtime_error("TensorShape: static element count overflow");
            }
            count *= d;
        }
        return count;
    }

    int64_t element_count(int64_t batch_size = 1) const {
        if (dims.empty()) return 0;
        if (is_dynamic()) {
            if (batch_size <= 0) {
                throw std::runtime_error("TensorShape: batch_size must be positive");
            }
            int64_t s = static_element_count();
            if (batch_size > (INT64_MAX / s)) {
                throw std::runtime_error("TensorShape: dynamic element count overflow");
            }
            return batch_size * s;
        }
        int64_t count = 1;
        for (auto d : dims) {
            if (d <= 0) {
                throw std::runtime_error("TensorShape: non-positive dimension encountered");
            }
            if (count > (INT64_MAX / d)) {
                throw std::runtime_error("TensorShape: element count overflow");
            }
            count *= d;
        }
        return count;
    }

    bool operator==(const TensorShape& other) const { return dims == other.dims; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) oss << ", ";
            if (dims[i] == DYNAMIC_DIM) {
                oss << "?";
            } else {
                oss << dims[i];
            }
        }
        oss << "]";
        return oss.str();
    }
};

// ============================================================================
// Tensor metadata (shape + dtype + layout)
// ============================================================================
struct TensorMeta {
    TensorShape shape;
    DType dtype = DType::FLOAT32;

    TensorMeta() = default;
    TensorMeta(TensorShape s, DType dt) : shape(std::move(s)), dtype(dt) {}

    bool is_dynamic() const { return shape.is_dynamic(); }
    int64_t element_count(int64_t batch_size = 1) const { return shape.element_count(batch_size); }
    int64_t static_element_count() const { return shape.static_element_count(); }

    size_t byte_size(int64_t batch_size = 1) const {
        return static_cast<size_t>(element_count(batch_size)) * dtype_size(dtype);
    }

    std::string to_string() const {
        return dtype_to_string(dtype) + shape.to_string();
    }
};

// ============================================================================
// IR operation kinds
// ============================================================================
enum class OpKind {
    INPUT,
    CONST,
    MATMUL,
    ADD,
    RELU,
    OUTPUT,
    FUSED_MATMUL_ADD_RELU,
    // ForgeTune v2.0 Operations
    MUL,
    SIGMOID,
    TANH,
    CONV2D,
    FLATTEN,
    FUSED_CONV2D_RELU,
};

inline const char* opkind_to_string(OpKind kind) {
    switch (kind) {
        case OpKind::INPUT:  return "INPUT";
        case OpKind::CONST:  return "CONST";
        case OpKind::MATMUL: return "MATMUL";
        case OpKind::ADD:    return "ADD";
        case OpKind::RELU:   return "RELU";
        case OpKind::OUTPUT: return "OUTPUT";
        case OpKind::FUSED_MATMUL_ADD_RELU: return "FUSED_MATMUL_ADD_RELU";
        case OpKind::MUL:    return "MUL";
        case OpKind::SIGMOID: return "SIGMOID";
        case OpKind::TANH:   return "TANH";
        case OpKind::CONV2D: return "CONV2D";
        case OpKind::FLATTEN: return "FLATTEN";
        case OpKind::FUSED_CONV2D_RELU: return "FUSED_CONV2D_RELU";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Convolution attributes for CONV2D
// ============================================================================
struct Conv2DAttrs {
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;
    int64_t groups = 1;
};

// ============================================================================
// IR Node: represents a single operation in the computation graph
// ============================================================================
using NodeId = int32_t;
constexpr NodeId INVALID_NODE_ID = -1;

struct IRNode {
    NodeId id = INVALID_NODE_ID;
    OpKind op = OpKind::INPUT;
    std::string name;                   // Human-readable name
    std::vector<NodeId> inputs;         // Operand node IDs
    TensorMeta output_meta;             // Output tensor metadata
    std::vector<float> const_data;      // For CONST nodes: the actual data
    Conv2DAttrs conv_attrs;             // For CONV2D nodes: convolution attributes
    bool deleted = false;               // Soft-delete flag for passes

    IRNode() = default;
    IRNode(NodeId id_in, OpKind op_in, std::string name_in, std::vector<NodeId> inputs_in, TensorMeta meta_in)
        : id(id_in), op(op_in), name(std::move(name_in)), inputs(std::move(inputs_in)),
          output_meta(std::move(meta_in)) {}

    // Validate expected arity (-1 indicates variable arity, e.g. CONV2D with optional bias)
    int expected_input_count() const {
        switch (op) {
            case OpKind::INPUT:  return 0;
            case OpKind::CONST:  return 0;
            case OpKind::MATMUL: return 2;
            case OpKind::ADD:    return 2;
            case OpKind::MUL:    return 2;
            case OpKind::RELU:   return 1;
            case OpKind::SIGMOID: return 1;
            case OpKind::TANH:   return 1;
            case OpKind::OUTPUT: return 1;
            case OpKind::FLATTEN: return 1;
            case OpKind::FUSED_MATMUL_ADD_RELU: return 3; // matrix, weight, bias
            case OpKind::CONV2D: return -1; // 2 or 3 inputs (x, w, optional bias)
            case OpKind::FUSED_CONV2D_RELU: return -1; // 2 or 3 inputs
            default: return -1;
        }
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "%" << id << " = " << opkind_to_string(op) << "(";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "%" << inputs[i];
        }
        oss << ") : " << output_meta.to_string();
        if (!name.empty()) {
            oss << "  // " << name;
        }
        return oss.str();
    }
};

// ============================================================================
// IRGraph: the complete computation graph
// ============================================================================
class IRGraph {
public:
    IRGraph() = default;

    // Add a node to the graph. Returns the node's ID.
    NodeId add_node(IRNode node) {
        if (node.id == INVALID_NODE_ID) {
            if (next_id_ >= std::numeric_limits<NodeId>::max()) {
                throw std::overflow_error("IRGraph::add_node: node ID at INT32_MAX cannot advance next_id_");
            }
            node.id = next_id_++;
        } else {
            if (node.id < 0) {
                throw std::runtime_error("IRGraph::add_node: negative node ID");
            }
            if (nodes_by_id_.count(node.id)) {
                throw std::runtime_error("IRGraph::add_node: duplicate node ID " +
                                         std::to_string(node.id));
            }
            if (node.id == std::numeric_limits<NodeId>::max()) {
                throw std::overflow_error("IRGraph::add_node: node ID at INT32_MAX cannot advance next_id_");
            }
            if (node.id >= next_id_) {
                next_id_ = node.id + 1;
            }
        }
        NodeId id = node.id;
        nodes_by_id_[id] = nodes_.size();
        nodes_.push_back(std::move(node));
        return id;
    }

    // Get a node by ID (const)
    const IRNode& get_node(NodeId id) const {
        auto it = nodes_by_id_.find(id);
        if (it == nodes_by_id_.end()) {
            throw std::runtime_error("IRGraph::get_node: unknown node ID " +
                                     std::to_string(id));
        }
        return nodes_[it->second];
    }

    // Get a mutable node by ID
    IRNode& get_node_mut(NodeId id) {
        auto it = nodes_by_id_.find(id);
        if (it == nodes_by_id_.end()) {
            throw std::runtime_error("IRGraph::get_node_mut: unknown node ID " +
                                     std::to_string(id));
        }
        return nodes_[it->second];
    }

    // Check if a node exists
    bool has_node(NodeId id) const {
        return nodes_by_id_.count(id) > 0;
    }

    // Set the graph input nodes
    void set_inputs(std::vector<NodeId> input_ids) {
        graph_inputs_ = std::move(input_ids);
    }

    // Set the graph output nodes
    void set_outputs(std::vector<NodeId> output_ids) {
        graph_outputs_ = std::move(output_ids);
    }

    const std::vector<NodeId>& get_inputs() const { return graph_inputs_; }
    const std::vector<NodeId>& get_outputs() const { return graph_outputs_; }

    // Get all nodes in insertion order (including deleted)
    const std::vector<IRNode>& nodes() const { return nodes_; }

    // Get all live (non-deleted) nodes in insertion order
    std::vector<const IRNode*> live_nodes() const {
        std::vector<const IRNode*> result;
        for (const auto& n : nodes_) {
            if (!n.deleted) result.push_back(&n);
        }
        return result;
    }

    // Topological order of live nodes (returns IDs)
    std::vector<NodeId> topological_order() const {
        std::vector<NodeId> order;
        std::unordered_set<NodeId> visited;
        std::unordered_set<NodeId> in_stack;

        std::function<void(NodeId)> visit = [&](NodeId id) {
            if (visited.count(id)) return;
            if (in_stack.count(id)) {
                throw std::runtime_error("IRGraph: cycle detected at node " +
                                         std::to_string(id));
            }
            const auto& node = get_node(id);
            if (node.deleted) return;

            in_stack.insert(id);
            for (NodeId inp : node.inputs) {
                visit(inp);
            }
            in_stack.erase(id);
            visited.insert(id);
            order.push_back(id);
        };

        for (const auto& n : nodes_) {
            if (!n.deleted) {
                visit(n.id);
            }
        }
        return order;
    }

    // Compute use counts for all live nodes
    std::unordered_map<NodeId, int> use_counts() const {
        std::unordered_map<NodeId, int> counts;
        for (const auto& n : nodes_) {
            if (n.deleted) continue;
            counts[n.id]; // ensure entry exists (may be 0)
            for (NodeId inp : n.inputs) {
                counts[inp]++;
            }
        }
        // Also count graph outputs
        for (NodeId out_id : graph_outputs_) {
            counts[out_id]++;
        }
        return counts;
    }

    // Consumer lookup: which live nodes consume a given node's output?
    std::unordered_map<NodeId, std::vector<NodeId>> consumer_map() const {
        std::unordered_map<NodeId, std::vector<NodeId>> consumers;
        for (const auto& n : nodes_) {
            if (n.deleted) continue;
            for (NodeId inp : n.inputs) {
                consumers[inp].push_back(n.id);
            }
        }
        return consumers;
    }

    // Soft-delete a node
    void delete_node(NodeId id) {
        auto& node = get_node_mut(id);
        node.deleted = true;
    }

    // Replace a node's inputs
    void replace_input(NodeId consumer_id, NodeId old_input, NodeId new_input) {
        auto& node = get_node_mut(consumer_id);
        for (auto& inp : node.inputs) {
            if (inp == old_input) {
                inp = new_input;
            }
        }
    }

    // Replace all uses of old_id with new_id in the entire graph
    void replace_all_uses(NodeId old_id, NodeId new_id) {
        for (auto& n : nodes_) {
            if (n.deleted) continue;
            for (auto& inp : n.inputs) {
                if (inp == old_id) inp = new_id;
            }
        }
        for (auto& out : graph_outputs_) {
            if (out == old_id) out = new_id;
        }
    }

    // Number of live nodes
    size_t live_node_count() const {
        size_t count = 0;
        for (const auto& n : nodes_) {
            if (!n.deleted) count++;
        }
        return count;
    }

    // Print the full IR
    std::string to_string() const {
        std::ostringstream oss;
        oss << "=== ForgeTune IR (v" << IR_SCHEMA_VERSION_MAJOR << "."
            << IR_SCHEMA_VERSION_MINOR << ") ===\n";
        oss << "Inputs: [";
        for (size_t i = 0; i < graph_inputs_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "%" << graph_inputs_[i];
        }
        oss << "]\n";
        oss << "Outputs: [";
        for (size_t i = 0; i < graph_outputs_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "%" << graph_outputs_[i];
        }
        oss << "]\n\n";

        for (const auto& n : nodes_) {
            if (n.deleted) continue;
            oss << "  " << n.to_string() << "\n";
        }
        oss << "=== End IR ===\n";
        return oss.str();
    }

    // Allocate a new unique node ID without adding a node
    NodeId allocate_id() {
        return next_id_++;
    }

    int64_t max_batch_size() const { return max_batch_size_; }
    void set_max_batch_size(int64_t mb) { max_batch_size_ = mb; }

private:
    std::vector<IRNode> nodes_;
    std::unordered_map<NodeId, size_t> nodes_by_id_; // ID -> index in nodes_
    std::vector<NodeId> graph_inputs_;
    std::vector<NodeId> graph_outputs_;
    NodeId next_id_ = 0;
    int64_t max_batch_size_ = 128;
};

// ============================================================================
// Error categories
// ============================================================================
class ForgeTuneError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class UnsupportedOpError : public ForgeTuneError {
public:
    UnsupportedOpError(const std::string& op_name)
        : ForgeTuneError("Unsupported operator: " + op_name), op_name_(op_name) {}
    const std::string& op_name() const { return op_name_; }
private:
    std::string op_name_;
};

class InvalidGraphError : public ForgeTuneError {
public:
    using ForgeTuneError::ForgeTuneError;
};

class InvalidShapeError : public ForgeTuneError {
public:
    using ForgeTuneError::ForgeTuneError;
};

class CodeGenError : public ForgeTuneError {
public:
    using ForgeTuneError::ForgeTuneError;
};

class CompilationError : public ForgeTuneError {
public:
    using ForgeTuneError::ForgeTuneError;
};

class CorrectnessError : public ForgeTuneError {
public:
    using ForgeTuneError::ForgeTuneError;
};

} // namespace forgetune

#endif // FORGETUNE_IR_HPP
