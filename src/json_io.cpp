#include "forgetune/json_io.hpp"
#include <sstream>
#include <iomanip>
#include <cctype>
#include <limits>

namespace forgetune {

namespace {
std::string escape_json_string(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        switch (c) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
                break;
        }
    }
    return ss.str();
}
} // anonymous namespace

std::string graph_to_json(const IRGraph& graph) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"schema_version_major\": " << IR_SCHEMA_VERSION_MAJOR << ",\n"
        << "  \"schema_version_minor\": " << IR_SCHEMA_VERSION_MINOR << ",\n"
        << "  \"max_batch_size\": " << graph.max_batch_size() << ",\n";

    // Inputs
    oss << "  \"inputs\": [";
    const auto& inputs = graph.get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << inputs[i];
    }
    oss << "],\n";

    // Outputs
    oss << "  \"outputs\": [";
    const auto& outputs = graph.get_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << outputs[i];
    }
    oss << "],\n";

    // Nodes
    oss << "  \"nodes\": [\n";
    const auto& nodes = graph.nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        oss << "    {\n"
            << "      \"id\": " << n.id << ",\n"
            << "      \"op\": \"" << opkind_to_string(n.op) << "\",\n"
            << "      \"name\": \"" << escape_json_string(n.name) << "\",\n"
            << "      \"deleted\": " << (n.deleted ? "true" : "false") << ",\n";

        if (n.op == OpKind::CONV2D || n.op == OpKind::FUSED_CONV2D_RELU) {
            oss << "      \"conv_attrs\": {"
                << "\"stride_h\": " << n.conv_attrs.stride_h << ", "
                << "\"stride_w\": " << n.conv_attrs.stride_w << ", "
                << "\"pad_h\": " << n.conv_attrs.pad_h << ", "
                << "\"pad_w\": " << n.conv_attrs.pad_w << ", "
                << "\"dilation_h\": " << n.conv_attrs.dilation_h << ", "
                << "\"dilation_w\": " << n.conv_attrs.dilation_w << ", "
                << "\"groups\": " << n.conv_attrs.groups << "},\n";
        }

        // Node inputs
        oss << "      \"inputs\": [";
        for (size_t j = 0; j < n.inputs.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << n.inputs[j];
        }
        oss << "],\n";

        // Shape
        oss << "      \"shape\": [";
        for (size_t j = 0; j < n.output_meta.shape.dims.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << n.output_meta.shape.dims[j];
        }
        oss << "],\n";

        // Dtype
        oss << "      \"dtype\": \"" << dtype_to_string(n.output_meta.dtype) << "\",\n";

        // Const data
        oss << "      \"const_data\": [";
        for (size_t j = 0; j < n.const_data.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << std::setprecision(std::numeric_limits<float>::max_digits10) << n.const_data[j];
        }
        oss << "]\n";

        oss << "    }" << (i + 1 < nodes.size() ? "," : "") << "\n";
    }
    oss << "  ]\n"
        << "}\n";

    return oss.str();
}

// Lightweight JSON token scanner
namespace {
    enum class TokType { LBRACE, RBRACE, LBRACKET, RBRACKET, COLON, COMMA, STRING, NUMBER, BOOLEAN, END_OF_FILE };

    struct Token {
        TokType type;
        std::string str_val;
        double num_val = 0.0;
        bool bool_val = false;
    };

    class JsonLexer {
    public:
        explicit JsonLexer(const std::string& src) : src_(src), pos_(0) {}

        Token next() {
            skip_whitespace();
            if (pos_ >= src_.size()) {
                return {TokType::END_OF_FILE, "", 0.0, false};
            }
            char c = src_[pos_++];
            switch (c) {
                case '{': return {TokType::LBRACE, "{", 0.0, false};
                case '}': return {TokType::RBRACE, "}", 0.0, false};
                case '[': return {TokType::LBRACKET, "[", 0.0, false};
                case ']': return {TokType::RBRACKET, "]", 0.0, false};
                case ':': return {TokType::COLON, ":", 0.0, false};
                case ',': return {TokType::COMMA, ",", 0.0, false};
                case '"': {
                    std::string s;
                    while (pos_ < src_.size()) {
                        char c2 = src_[pos_++];
                        if (c2 == '"') {
                            return {TokType::STRING, s, 0.0, false};
                        }
                        if (c2 == '\\') {
                            if (pos_ >= src_.size()) {
                                throw InvalidGraphError("JSON Lexer: Unterminated escape sequence at end of string");
                            }
                            char esc = src_[pos_++];
                            switch (esc) {
                                case '"':  s += '"'; break;
                                case '\\': s += '\\'; break;
                                case '/':  s += '/'; break;
                                case 'b':  s += '\b'; break;
                                case 'f':  s += '\f'; break;
                                case 'n':  s += '\n'; break;
                                case 'r':  s += '\r'; break;
                                case 't':  s += '\t'; break;
                                case 'u': {
                                    if (pos_ + 4 > src_.size()) {
                                        throw InvalidGraphError("JSON Lexer: Unterminated \\u hex escape in string");
                                    }
                                    std::string hex_str = src_.substr(pos_, 4);
                                    pos_ += 4;
                                    char* endptr = nullptr;
                                    unsigned long cp = std::strtoul(hex_str.c_str(), &endptr, 16);
                                    if (endptr != hex_str.c_str() + 4) {
                                        throw InvalidGraphError("JSON Lexer: Invalid \\u hex digits: " + hex_str);
                                    }
                                    if (cp <= 0x7F) {
                                        s += static_cast<char>(cp);
                                    } else if (cp <= 0x7FF) {
                                        s += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                                        s += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else {
                                        s += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                                        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                        s += static_cast<char>(0x80 | (cp & 0x3F));
                                    }
                                    break;
                                }
                                default:
                                    throw InvalidGraphError(std::string("JSON Lexer: Invalid escape character '\\") + esc + "'");
                            }
                        } else {
                            s += c2;
                        }
                    }
                    throw InvalidGraphError("JSON Lexer: Unterminated string literal");
                }
                default: {
                    if (c == '-' || std::isdigit(c)) {
                        size_t start = pos_ - 1;
                        while (pos_ < src_.size() && (std::isdigit(src_[pos_]) || src_[pos_] == '.' ||
                                                      src_[pos_] == 'e' || src_[pos_] == 'E' ||
                                                      src_[pos_] == '+' || src_[pos_] == '-')) {
                            pos_++;
                        }
                        std::string num_str = src_.substr(start, pos_ - start);
                        double val = 0.0;
                        try {
                            size_t idx = 0;
                            val = std::stod(num_str, &idx);
                            if (idx != num_str.size()) {
                                throw InvalidGraphError("JSON Lexer: Invalid number format encountered: '" + num_str + "'");
                            }
                        } catch (const InvalidGraphError&) {
                            throw;
                        } catch (const std::exception& e) {
                            throw InvalidGraphError("JSON Lexer: Invalid number format encountered: '" + num_str + "'");
                        }
                        return {TokType::NUMBER, num_str, val, false};
                    } else if (std::isalpha(c)) {
                        size_t start = pos_ - 1;
                        while (pos_ < src_.size() && std::isalpha(src_[pos_])) {
                            pos_++;
                        }
                        std::string word = src_.substr(start, pos_ - start);
                        if (word == "true") return {TokType::BOOLEAN, "true", 0.0, true};
                        if (word == "false") return {TokType::BOOLEAN, "false", 0.0, false};
                    }
                    break;
                }
            }
            return {TokType::END_OF_FILE, "", 0.0, false};
        }

    private:
        void skip_whitespace() {
            while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t' ||
                                          src_[pos_] == '\n' || src_[pos_] == '\r')) {
                pos_++;
            }
        }
        const std::string& src_;
        size_t pos_;
    };
} // anonymous namespace

IRGraph json_to_graph(const std::string& json_str) {
    IRGraph graph;
    JsonLexer lexer(json_str);

    // Simple top-down parse of graph JSON
    Token t = lexer.next();
    if (t.type != TokType::LBRACE) {
        throw InvalidGraphError("json_to_graph: expected '{' at start of JSON");
    }

    std::vector<NodeId> graph_inputs;
    std::vector<NodeId> graph_outputs;
    bool closed = false;

    while (true) {
        t = lexer.next();
        if (t.type == TokType::RBRACE) {
            closed = true;
            break;
        }
        if (t.type == TokType::END_OF_FILE) break;
        if (t.type == TokType::COMMA) continue;
        if (t.type != TokType::STRING) continue;

        std::string key = t.str_val;
        Token colon = lexer.next(); // Consume colon
        if (colon.type == TokType::END_OF_FILE) {
            throw InvalidGraphError("json_to_graph: Unexpected EOF after key '" + key + "'");
        }

        if (key == "schema_version_major") {
            Token v_tok = lexer.next();
            if (v_tok.type == TokType::NUMBER) {
                int64_t v = static_cast<int64_t>(v_tok.num_val);
                if (v < 1 || v > IR_SCHEMA_VERSION_MAJOR) {
                    throw InvalidGraphError("json_to_graph: Unsupported schema version major: " + std::to_string(v) + " (supported: 1 to " + std::to_string(IR_SCHEMA_VERSION_MAJOR) + ")");
                }
            } else if (v_tok.type == TokType::END_OF_FILE) {
                throw InvalidGraphError("json_to_graph: Unexpected EOF reading schema_version_major");
            }
        } else if (key == "max_batch_size") {
            Token mb_tok = lexer.next();
            if (mb_tok.type == TokType::NUMBER) {
                int64_t mb = static_cast<int64_t>(mb_tok.num_val);
                if (mb <= 0) {
                    throw InvalidGraphError("json_to_graph: max_batch_size must be positive, got " + std::to_string(mb));
                }
                graph.set_max_batch_size(mb);
            } else if (mb_tok.type == TokType::END_OF_FILE) {
                throw InvalidGraphError("json_to_graph: Unexpected EOF reading max_batch_size");
            }
        } else if (key == "inputs") {
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF at inputs '['");
            while (true) {
                Token in_tok = lexer.next();
                if (in_tok.type == TokType::RBRACKET) break;
                if (in_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing inputs array");
                if (in_tok.type == TokType::NUMBER) {
                    graph_inputs.push_back(static_cast<NodeId>(in_tok.num_val));
                }
            }
        } else if (key == "outputs") {
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF at outputs '['");
            while (true) {
                Token out_tok = lexer.next();
                if (out_tok.type == TokType::RBRACKET) break;
                if (out_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing outputs array");
                if (out_tok.type == TokType::NUMBER) {
                    graph_outputs.push_back(static_cast<NodeId>(out_tok.num_val));
                }
            }
        } else if (key == "nodes") {
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF at nodes '['");
            while (true) {
                Token node_tok = lexer.next();
                if (node_tok.type == TokType::RBRACKET) break;
                if (node_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing nodes array");
                if (node_tok.type == TokType::COMMA) continue;
                if (node_tok.type != TokType::LBRACE) continue;

                IRNode node;
                while (true) {
                    Token f_key = lexer.next();
                    if (f_key.type == TokType::RBRACE) break;
                    if (f_key.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing node fields");
                    if (f_key.type == TokType::COMMA) continue;
                    if (f_key.type != TokType::STRING) continue;

                    lexer.next(); // Consume colon
                    std::string field = f_key.str_val;

                    if (field == "id") {
                        Token id_tok = lexer.next();
                        if (id_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading node id");
                        node.id = static_cast<NodeId>(id_tok.num_val);
                    } else if (field == "op") {
                        Token op_tok = lexer.next();
                        if (op_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading node op");
                        std::string op_str = op_tok.str_val;
                        if (op_str == "INPUT") node.op = OpKind::INPUT;
                        else if (op_str == "CONST") node.op = OpKind::CONST;
                        else if (op_str == "MATMUL") node.op = OpKind::MATMUL;
                        else if (op_str == "ADD") node.op = OpKind::ADD;
                        else if (op_str == "RELU") node.op = OpKind::RELU;
                        else if (op_str == "MUL") node.op = OpKind::MUL;
                        else if (op_str == "SIGMOID") node.op = OpKind::SIGMOID;
                        else if (op_str == "TANH") node.op = OpKind::TANH;
                        else if (op_str == "FLATTEN") node.op = OpKind::FLATTEN;
                        else if (op_str == "CONV2D") node.op = OpKind::CONV2D;
                        else if (op_str == "FUSED_CONV2D_RELU") node.op = OpKind::FUSED_CONV2D_RELU;
                        else if (op_str == "OUTPUT") node.op = OpKind::OUTPUT;
                        else if (op_str == "FUSED_MATMUL_ADD_RELU") node.op = OpKind::FUSED_MATMUL_ADD_RELU;
                        else throw UnsupportedOpError(op_str);
                    } else if (field == "conv_attrs") {
                        Token ca_tok = lexer.next(); // '{'
                        if (ca_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF at conv_attrs '{'");
                        while (true) {
                            Token a_key = lexer.next();
                            if (a_key.type == TokType::RBRACE) break;
                            if (a_key.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing conv_attrs");
                            if (a_key.type == TokType::COMMA) continue;
                            if (a_key.type != TokType::STRING) continue;
                            lexer.next(); // ':'
                            std::string attr_name = a_key.str_val;
                            Token attr_val = lexer.next();
                            if (attr_val.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading conv_attr value");
                            int64_t v = static_cast<int64_t>(attr_val.num_val);
                            if (attr_name == "stride_h") node.conv_attrs.stride_h = v;
                            else if (attr_name == "stride_w") node.conv_attrs.stride_w = v;
                            else if (attr_name == "pad_h") node.conv_attrs.pad_h = v;
                            else if (attr_name == "pad_w") node.conv_attrs.pad_w = v;
                            else if (attr_name == "dilation_h") node.conv_attrs.dilation_h = v;
                            else if (attr_name == "dilation_w") node.conv_attrs.dilation_w = v;
                            else if (attr_name == "groups") node.conv_attrs.groups = v;
                        }
                    } else if (field == "name") {
                        Token name_tok = lexer.next();
                        if (name_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading node name");
                        node.name = name_tok.str_val;
                    } else if (field == "deleted") {
                        Token del_tok = lexer.next();
                        if (del_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading node deleted");
                        node.deleted = del_tok.bool_val;
                    } else if (field == "inputs") {
                        Token in_bracket = lexer.next(); // '['
                        if (in_bracket.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF at node inputs '['");
                        while (true) {
                            Token inp_id_tok = lexer.next();
                            if (inp_id_tok.type == TokType::RBRACKET) break;
                            if (inp_id_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing node inputs");
                            if (inp_id_tok.type == TokType::NUMBER) {
                                node.inputs.push_back(static_cast<NodeId>(inp_id_tok.num_val));
                            }
                        }
                    } else if (field == "shape") {
                        Token sh_bracket = lexer.next(); // '['
                        if (sh_bracket.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF at node shape '['");
                        std::vector<int64_t> dims;
                        while (true) {
                            Token dim_tok = lexer.next();
                            if (dim_tok.type == TokType::RBRACKET) break;
                            if (dim_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing node shape");
                            if (dim_tok.type == TokType::NUMBER) {
                                dims.push_back(static_cast<int64_t>(dim_tok.num_val));
                            }
                        }
                        node.output_meta.shape = TensorShape(std::move(dims));
                    } else if (field == "dtype") {
                        Token dt_tok = lexer.next(); // string
                        if (dt_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF reading node dtype");
                        if (dt_tok.str_val != "f32" && dt_tok.str_val != "float32") {
                            throw InvalidGraphError("json_to_graph: unsupported dtype: '" + dt_tok.str_val + "' (only f32/float32 supported)");
                        }
                        node.output_meta.dtype = DType::FLOAT32;
                    } else if (field == "const_data") {
                        Token cd_bracket = lexer.next(); // '['
                        if (cd_bracket.type == TokType::END_OF_FILE) throw InvalidGraphError("Unexpected EOF at node const_data '['");
                        while (true) {
                            Token num_tok = lexer.next();
                            if (num_tok.type == TokType::RBRACKET) break;
                            if (num_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF while parsing node const_data");
                            if (num_tok.type == TokType::NUMBER) {
                                node.const_data.push_back(static_cast<float>(num_tok.num_val));
                            }
                        }
                    } else {
                        // Skip unknown field
                        lexer.next();
                    }
                }
                graph.add_node(std::move(node));
            }
        } else {
            // Skip other fields
            lexer.next();
        }
    }

    if (!closed) {
        throw InvalidGraphError("json_to_graph: Unexpected EOF before closing '}' of graph object");
    }

    graph.set_inputs(std::move(graph_inputs));
    graph.set_outputs(std::move(graph_outputs));

    return graph;
}

} // namespace forgetune
