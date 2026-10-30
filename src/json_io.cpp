#include "forgetune/json_io.hpp"
#include <sstream>
#include <iomanip>
#include <cctype>
#include <cmath>
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
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(c));
                } else {
                    ss << c;
                }
                break;
        }
    }
    return ss.str();
}

bool is_valid_json_number(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[i] == '-') {
        i++;
        if (i >= s.size()) return false;
    }
    // Integer part
    if (s[i] == '0') {
        i++;
        // Leading zero cannot be followed by another digit (e.g. "01" is invalid)
        if (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    } else if (std::isdigit(static_cast<unsigned char>(s[i]))) {
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    } else {
        return false;
    }
    // Fraction part
    if (i < s.size() && s[i] == '.') {
        i++;
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false; // "1." is invalid
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    }
    // Exponent part
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    }
    return i == s.size();
}

// Forward declarations for strict JSON integer parsing
enum class TokType { LBRACE, RBRACE, LBRACKET, RBRACKET, COLON, COMMA, STRING, NUMBER, BOOLEAN, NULL_VAL, END_OF_FILE };

struct Token {
    TokType type;
    std::string str_val;
    double num_val = 0.0;
    bool bool_val = false;
};

int64_t parse_json_integer(const Token& tok, const std::string& field_name, int64_t min_val = std::numeric_limits<int64_t>::min(), int64_t max_val = std::numeric_limits<int64_t>::max()) {
    if (tok.type != TokType::NUMBER) {
        throw InvalidGraphError("json_to_graph: field '" + field_name + "' must be a number");
    }
    const std::string& s = tok.str_val;
    if (s.find('.') != std::string::npos || s.find('e') != std::string::npos || s.find('E') != std::string::npos) {
        throw InvalidGraphError("json_to_graph: field '" + field_name + "' must be an integer, got: " + s);
    }
    try {
        size_t idx = 0;
        long long val = std::stoll(s, &idx);
        if (idx != s.size()) {
            throw InvalidGraphError("json_to_graph: field '" + field_name + "' has invalid integer syntax: " + s);
        }
        if (val < min_val || val > max_val) {
            throw InvalidGraphError("json_to_graph: field '" + field_name + "' out of valid range [" + std::to_string(min_val) + ", " + std::to_string(max_val) + "]: " + s);
        }
        return static_cast<int64_t>(val);
    } catch (const std::out_of_range&) {
        throw InvalidGraphError("json_to_graph: field '" + field_name + "' out of valid 64-bit integer range: " + s);
    } catch (const std::invalid_argument&) {
        throw InvalidGraphError("json_to_graph: field '" + field_name + "' cannot be parsed as integer: " + s);
    }
}

OpKind string_to_opkind(const std::string& op_str) {
    if (op_str == "INPUT") return OpKind::INPUT;
    if (op_str == "CONST" || op_str == "CONSTANT") return OpKind::CONST;
    if (op_str == "MATMUL") return OpKind::MATMUL;
    if (op_str == "ADD") return OpKind::ADD;
    if (op_str == "RELU") return OpKind::RELU;
    if (op_str == "MUL") return OpKind::MUL;
    if (op_str == "SIGMOID") return OpKind::SIGMOID;
    if (op_str == "TANH") return OpKind::TANH;
    if (op_str == "CONV2D") return OpKind::CONV2D;
    if (op_str == "FLATTEN") return OpKind::FLATTEN;
    if (op_str == "OUTPUT") return OpKind::OUTPUT;
    if (op_str == "FUSED_MATMUL_ADD_RELU" || op_str == "FUSED_MATMUL_BIAS_RELU") return OpKind::FUSED_MATMUL_ADD_RELU;
    if (op_str == "FUSED_CONV2D_RELU") return OpKind::FUSED_CONV2D_RELU;
    throw InvalidGraphError("json_to_graph: Unknown op kind '" + op_str + "'");
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
            float f = n.const_data[j];
            if (!std::isfinite(f)) {
                throw InvalidGraphError("graph_to_json: non-finite float32 in const_data");
            }
            oss << std::setprecision(std::numeric_limits<float>::max_digits10) << f;
        }
        oss << "]\n";

        oss << "    }" << (i + 1 < nodes.size() ? "," : "") << "\n";
    }
    oss << "  ]\n"
        << "}\n";

    return oss.str();
}

// Strict JSON token scanner
namespace {
    class JsonLexer {
    public:
        explicit JsonLexer(const std::string& src) : src_(src), pos_(0) {}

        void skip_whitespace() {
            while (pos_ < src_.size()) {
                unsigned char c = static_cast<unsigned char>(src_[pos_]);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    pos_++;
                } else {
                    break;
                }
            }
        }

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
                        if (s.size() > 4096) {
                            throw InvalidGraphError("JSON Lexer: String literal exceeds maximum allowed length (4096 bytes)");
                        }
                        // RFC 8259 Section 7: Unescaped control characters (< 0x20) are strictly forbidden
                        if (static_cast<unsigned char>(c2) < 0x20) {
                            throw InvalidGraphError("JSON Lexer: Unescaped control character in string literal (ASCII " + std::to_string(static_cast<int>(static_cast<unsigned char>(c2))) + ")");
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
                                    for (char hc : hex_str) {
                                        if (!std::isxdigit(static_cast<unsigned char>(hc))) {
                                            throw InvalidGraphError("JSON Lexer: Invalid \\u hex digits: " + hex_str);
                                        }
                                    }
                                    pos_ += 4;
                                    char* endptr = nullptr;
                                    unsigned long cp = std::strtoul(hex_str.c_str(), &endptr, 16);
                                    if (endptr != hex_str.c_str() + 4) {
                                        throw InvalidGraphError("JSON Lexer: Invalid \\u hex digits: " + hex_str);
                                    }

                                    // Unicode surrogate pair handling (RFC 8259 / UTF-16 surrogate pairs)
                                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                                        // High surrogate: must be immediately followed by \uDC00..\uDFFF (low surrogate)
                                        if (pos_ + 6 <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                                            std::string low_hex = src_.substr(pos_ + 2, 4);
                                            for (char lc : low_hex) {
                                                if (!std::isxdigit(static_cast<unsigned char>(lc))) {
                                                    throw InvalidGraphError("JSON Lexer: Invalid \\u hex digits: " + low_hex);
                                                }
                                            }
                                            char* low_end = nullptr;
                                            unsigned long low_cp = std::strtoul(low_hex.c_str(), &low_end, 16);
                                            if (low_end == low_hex.c_str() + 4 && low_cp >= 0xDC00 && low_cp <= 0xDFFF) {
                                                pos_ += 6;
                                                cp = 0x10000 + (((cp - 0xD800) << 10) | (low_cp - 0xDC00));
                                            } else {
                                                throw InvalidGraphError("JSON Lexer: Isolated high surrogate \\u" + hex_str);
                                            }
                                        } else {
                                            throw InvalidGraphError("JSON Lexer: Isolated high surrogate \\u" + hex_str);
                                        }
                                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                                        throw InvalidGraphError("JSON Lexer: Isolated low surrogate \\u" + hex_str);
                                    }

                                    // Encode code point into valid UTF-8
                                    if (cp <= 0x7F) {
                                        s += static_cast<char>(cp);
                                    } else if (cp <= 0x7FF) {
                                        s += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                                        s += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else if (cp <= 0xFFFF) {
                                        s += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                                        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                        s += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else if (cp <= 0x10FFFF) {
                                        // 4-byte UTF-8 for supplementary code point (e.g. emoji 😀)
                                        s += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
                                        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                                        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                        s += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else {
                                        throw InvalidGraphError("JSON Lexer: Unicode code point out of range: " + std::to_string(cp));
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
                    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
                        size_t start = pos_ - 1;
                        while (pos_ < src_.size() && (std::isdigit(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '.' ||
                                                      src_[pos_] == 'e' || src_[pos_] == 'E' ||
                                                      src_[pos_] == '+' || src_[pos_] == '-')) {
                            pos_++;
                        }
                        std::string num_str = src_.substr(start, pos_ - start);
                        if (!is_valid_json_number(num_str)) {
                            throw InvalidGraphError("JSON Lexer: Invalid JSON number grammar: '" + num_str + "'");
                        }
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
                    } else if (std::isalpha(static_cast<unsigned char>(c))) {
                        size_t start = pos_ - 1;
                        while (pos_ < src_.size() && std::isalpha(static_cast<unsigned char>(src_[pos_]))) {
                            pos_++;
                        }
                        std::string word = src_.substr(start, pos_ - start);
                        if (word == "true") return {TokType::BOOLEAN, word, 1.0, true};
                        if (word == "false") return {TokType::BOOLEAN, word, 0.0, false};
                        if (word == "null") return {TokType::NULL_VAL, word, 0.0, false};
                        throw InvalidGraphError("JSON Lexer: Unexpected identifier: '" + word + "'");
                    }
                    throw InvalidGraphError(std::string("JSON Lexer: Unexpected character '") + c + "'");
                }
            }
        }

        void skip_value(Token t) {
            if (t.type == TokType::LBRACE) {
                bool first = true;
                while (true) {
                    Token cur = next();
                    if (cur.type == TokType::RBRACE) break;
                    if (!first) {
                        if (cur.type != TokType::COMMA) {
                            throw InvalidGraphError("json_to_graph: expected ',' between nested object properties");
                        }
                        cur = next();
                    }
                    first = false;
                    if (cur.type == TokType::END_OF_FILE) {
                        throw InvalidGraphError("json_to_graph: Unexpected EOF in nested object");
                    }
                    if (cur.type != TokType::STRING) {
                        throw InvalidGraphError("json_to_graph: expected string key in nested object");
                    }
                    Token col = next();
                    if (col.type != TokType::COLON) {
                        throw InvalidGraphError("json_to_graph: expected ':' after key in nested object");
                    }
                    skip_value(next());
                }
            } else if (t.type == TokType::LBRACKET) {
                bool first = true;
                while (true) {
                    Token cur = next();
                    if (cur.type == TokType::RBRACKET) break;
                    if (!first) {
                        if (cur.type != TokType::COMMA) {
                            throw InvalidGraphError("json_to_graph: expected ',' between nested array elements");
                        }
                        cur = next();
                    }
                    first = false;
                    if (cur.type == TokType::END_OF_FILE) {
                        throw InvalidGraphError("json_to_graph: Unexpected EOF in nested array");
                    }
                    skip_value(cur);
                }
            } else if (t.type == TokType::STRING || t.type == TokType::NUMBER ||
                       t.type == TokType::BOOLEAN || t.type == TokType::NULL_VAL) {
                // Valid scalar value
            } else {
                throw InvalidGraphError("json_to_graph: unexpected token while parsing JSON value: " + std::to_string(static_cast<int>(t.type)));
            }
        }

    private:
        const std::string& src_;
        size_t pos_;
    };
} // anonymous namespace

IRGraph json_to_graph(const std::string& json_str) {
    IRGraph graph;
    JsonLexer lexer(json_str);

    // Top-down parse of graph JSON
    Token t = lexer.next();
    if (t.type != TokType::LBRACE) {
        throw InvalidGraphError("json_to_graph: expected '{' at start of JSON");
    }

    std::vector<NodeId> graph_inputs;
    std::vector<NodeId> graph_outputs;
    bool closed = false;
    bool first_prop = true;
    bool has_schema_major = false;
    bool has_inputs = false;
    bool has_outputs = false;
    bool has_nodes = false;
    std::unordered_set<std::string> seen_root_keys;

    while (true) {
        t = lexer.next();
        if (t.type == TokType::RBRACE) {
            closed = true;
            break;
        }
        if (t.type == TokType::END_OF_FILE) break;
        if (!first_prop) {
            if (t.type != TokType::COMMA) {
                throw InvalidGraphError("json_to_graph: expected ',' between root object properties");
            }
            t = lexer.next();
        }
        first_prop = false;
        if (t.type != TokType::STRING) {
            throw InvalidGraphError("json_to_graph: expected string key in graph object, got token " + std::to_string(static_cast<int>(t.type)));
        }

        std::string key = t.str_val;
        if (!seen_root_keys.insert(key).second) {
            throw InvalidGraphError("json_to_graph: duplicate property '" + key + "' in root object");
        }
        Token colon = lexer.next(); // Strictly require colon
        if (colon.type != TokType::COLON) {
            throw InvalidGraphError("json_to_graph: expected ':' after key '" + key + "'");
        }

        if (key == "schema_version_major") {
            has_schema_major = true;
            Token v_tok = lexer.next();
            if (v_tok.type == TokType::NUMBER) {
                int64_t v = parse_json_integer(v_tok, "schema_version_major", 1, IR_SCHEMA_VERSION_MAJOR);
                (void)v;
            } else {
                throw InvalidGraphError("json_to_graph: schema_version_major must be a number");
            }
        } else if (key == "schema_version_minor") {
            Token v_tok = lexer.next();
            if (v_tok.type == TokType::NUMBER) {
                parse_json_integer(v_tok, "schema_version_minor", 0, IR_SCHEMA_VERSION_MINOR);
            } else {
                throw InvalidGraphError("json_to_graph: schema_version_minor must be a number");
            }
        } else if (key == "max_batch_size") {
            Token mb_tok = lexer.next();
            if (mb_tok.type == TokType::NUMBER) {
                int64_t mb = parse_json_integer(mb_tok, "max_batch_size", 1, 1000000);
                graph.set_max_batch_size(mb);
            } else {
                throw InvalidGraphError("json_to_graph: max_batch_size must be a number");
            }
        } else if (key == "inputs") {
            has_inputs = true;
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for inputs");
            bool first_in = true;
            while (true) {
                Token in_tok = lexer.next();
                if (in_tok.type == TokType::RBRACKET) break;
                if (!first_in) {
                    if (in_tok.type != TokType::COMMA) {
                        throw InvalidGraphError("json_to_graph: expected ',' between inputs array elements");
                    }
                    in_tok = lexer.next();
                }
                first_in = false;
                if (in_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in inputs array");
                if (in_tok.type == TokType::NUMBER) {
                    graph_inputs.push_back(static_cast<NodeId>(parse_json_integer(in_tok, "input_id", 0)));
                } else {
                    throw InvalidGraphError("json_to_graph: input node ID must be a non-negative integer");
                }
            }
        } else if (key == "outputs") {
            has_outputs = true;
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for outputs");
            bool first_out = true;
            while (true) {
                Token out_tok = lexer.next();
                if (out_tok.type == TokType::RBRACKET) break;
                if (!first_out) {
                    if (out_tok.type != TokType::COMMA) {
                        throw InvalidGraphError("json_to_graph: expected ',' between outputs array elements");
                    }
                    out_tok = lexer.next();
                }
                first_out = false;
                if (out_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in outputs array");
                if (out_tok.type == TokType::NUMBER) {
                    graph_outputs.push_back(static_cast<NodeId>(parse_json_integer(out_tok, "output_id", 0)));
                } else {
                    throw InvalidGraphError("json_to_graph: output node ID must be a non-negative integer");
                }
            }
        } else if (key == "nodes") {
            has_nodes = true;
            Token lbrk = lexer.next(); // Consume '['
            if (lbrk.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for nodes");
            bool first_node = true;
            while (true) {
                Token n_tok = lexer.next();
                if (n_tok.type == TokType::RBRACKET) break;
                if (!first_node) {
                    if (n_tok.type != TokType::COMMA) {
                        throw InvalidGraphError("json_to_graph: expected ',' between nodes array elements");
                    }
                    n_tok = lexer.next();
                }
                first_node = false;
                if (n_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in nodes array");
                if (n_tok.type != TokType::LBRACE) {
                    throw InvalidGraphError("json_to_graph: expected '{' for node in nodes array");
                }

                IRNode node;
                bool node_closed = false;
                bool has_id = false, has_op = false, has_name = false, has_shape = false, has_dtype = false;
                bool first_field = true;
                std::unordered_set<std::string> seen_node_keys;
                while (true) {
                    Token f_tok = lexer.next();
                    if (f_tok.type == TokType::RBRACE) {
                        node_closed = true;
                        break;
                    }
                    if (f_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in node object");
                    if (!first_field) {
                        if (f_tok.type != TokType::COMMA) {
                            throw InvalidGraphError("json_to_graph: expected ',' between node properties");
                        }
                        f_tok = lexer.next();
                    }
                    first_field = false;
                    if (f_tok.type != TokType::STRING) {
                        throw InvalidGraphError("json_to_graph: expected string property name in node object");
                    }

                    std::string f_key = f_tok.str_val;
                    if (!seen_node_keys.insert(f_key).second) {
                        throw InvalidGraphError("json_to_graph: duplicate property '" + f_key + "' in node object");
                    }
                    Token f_colon = lexer.next();
                    if (f_colon.type != TokType::COLON) {
                        throw InvalidGraphError("json_to_graph: expected ':' after node property '" + f_key + "'");
                    }

                    if (f_key == "id") {
                        Token id_tok = lexer.next();
                        node.id = static_cast<NodeId>(parse_json_integer(id_tok, "node.id", 0));
                        has_id = true;
                    } else if (f_key == "op") {
                        Token op_tok = lexer.next();
                        if (op_tok.type != TokType::STRING) {
                            throw InvalidGraphError("json_to_graph: node.op must be a string");
                        }
                        node.op = string_to_opkind(op_tok.str_val);
                        has_op = true;
                    } else if (f_key == "name") {
                        Token name_tok = lexer.next();
                        if (name_tok.type != TokType::STRING) {
                            throw InvalidGraphError("json_to_graph: node.name must be a string");
                        }
                        node.name = name_tok.str_val;
                        has_name = true;
                    } else if (f_key == "deleted") {
                        Token del_tok = lexer.next();
                        if (del_tok.type != TokType::BOOLEAN) {
                            throw InvalidGraphError("json_to_graph: node.deleted must be a boolean");
                        }
                        node.deleted = del_tok.bool_val;
                    } else if (f_key == "conv_attrs") {
                        Token ca_tok = lexer.next();
                        if (ca_tok.type != TokType::LBRACE) {
                            throw InvalidGraphError("json_to_graph: expected '{' for conv_attrs");
                        }
                        bool first_ca = true;
                        std::unordered_set<std::string> seen_ca_keys;
                        while (true) {
                            Token c_f = lexer.next();
                            if (c_f.type == TokType::RBRACE) break;
                            if (!first_ca) {
                                if (c_f.type != TokType::COMMA) {
                                    throw InvalidGraphError("json_to_graph: expected ',' between conv_attrs properties");
                                }
                                c_f = lexer.next();
                            }
                            first_ca = false;
                            if (c_f.type == TokType::END_OF_FILE) {
                                throw InvalidGraphError("json_to_graph: Unexpected EOF in conv_attrs");
                            }
                            if (c_f.type != TokType::STRING) {
                                throw InvalidGraphError("json_to_graph: expected string key in conv_attrs");
                            }
                            std::string ck = c_f.str_val;
                            if (!seen_ca_keys.insert(ck).second) {
                                throw InvalidGraphError("json_to_graph: duplicate key in conv_attrs: '" + ck + "'");
                            }
                            Token cc = lexer.next();
                            if (cc.type != TokType::COLON) {
                                throw InvalidGraphError("json_to_graph: expected ':' after conv_attrs key '" + ck + "'");
                            }
                            Token cv = lexer.next();
                            if (ck == "stride_h") {
                                node.conv_attrs.stride_h = parse_json_integer(cv, ck, 1, 4096);
                            } else if (ck == "stride_w") {
                                node.conv_attrs.stride_w = parse_json_integer(cv, ck, 1, 4096);
                            } else if (ck == "pad_h") {
                                node.conv_attrs.pad_h = parse_json_integer(cv, ck, 0, 4096);
                            } else if (ck == "pad_w") {
                                node.conv_attrs.pad_w = parse_json_integer(cv, ck, 0, 4096);
                            } else if (ck == "dilation_h") {
                                node.conv_attrs.dilation_h = parse_json_integer(cv, ck, 1, 4096);
                            } else if (ck == "dilation_w") {
                                node.conv_attrs.dilation_w = parse_json_integer(cv, ck, 1, 4096);
                            } else if (ck == "groups") {
                                node.conv_attrs.groups = parse_json_integer(cv, ck, 1, 1);
                            } else {
                                throw InvalidGraphError("json_to_graph: unknown key in conv_attrs: '" + ck + "'");
                            }
                        }
                    } else if (f_key == "inputs") {
                        Token in_arr = lexer.next();
                        if (in_arr.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for node inputs");
                        bool first_nid = true;
                        while (true) {
                            Token id_tok = lexer.next();
                            if (id_tok.type == TokType::RBRACKET) break;
                            if (!first_nid) {
                                if (id_tok.type != TokType::COMMA) {
                                    throw InvalidGraphError("json_to_graph: expected ',' between node inputs elements");
                                }
                                id_tok = lexer.next();
                            }
                            first_nid = false;
                            if (id_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in node inputs array");
                            node.inputs.push_back(static_cast<NodeId>(parse_json_integer(id_tok, "node_input_id", 0)));
                        }
                    } else if (f_key == "shape") {
                        Token sh_arr = lexer.next();
                        if (sh_arr.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for node shape");
                        std::vector<int64_t> dims;
                        bool first_dim = true;
                        while (true) {
                            Token dim_tok = lexer.next();
                            if (dim_tok.type == TokType::RBRACKET) break;
                            if (!first_dim) {
                                if (dim_tok.type != TokType::COMMA) {
                                    throw InvalidGraphError("json_to_graph: expected ',' between shape dimensions");
                                }
                                dim_tok = lexer.next();
                            }
                            first_dim = false;
                            if (dim_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in node shape array");
                            dims.push_back(parse_json_integer(dim_tok, "dim"));
                        }
                        node.output_meta.shape = TensorShape(dims);
                        has_shape = true;
                    } else if (f_key == "dtype") {
                        Token dt_tok = lexer.next();
                        if (dt_tok.type != TokType::STRING) {
                            throw InvalidGraphError("json_to_graph: node.dtype must be a string");
                        }
                        std::string dt_lower = dt_tok.str_val;
                        std::transform(dt_lower.begin(), dt_lower.end(), dt_lower.begin(), [](unsigned char c) { return std::tolower(c); });
                        if (dt_lower != "f32" && dt_lower != "float32") {
                            throw InvalidGraphError("json_to_graph: Unsupported dtype '" + dt_tok.str_val + "', only 'f32'/'float32' supported in ForgeTune v2.0");
                        }
                        node.output_meta.dtype = DType::FLOAT32;
                        has_dtype = true;
                    } else if (f_key == "const_data") {
                        Token cd_arr = lexer.next();
                        if (cd_arr.type != TokType::LBRACKET) throw InvalidGraphError("json_to_graph: expected '[' for const_data");
                        bool first_c = true;
                        while (true) {
                            Token num_tok = lexer.next();
                            if (num_tok.type == TokType::RBRACKET) break;
                            if (!first_c) {
                                if (num_tok.type != TokType::COMMA) {
                                    throw InvalidGraphError("json_to_graph: expected ',' between const_data elements");
                                }
                                num_tok = lexer.next();
                            }
                            first_c = false;
                            if (num_tok.type == TokType::END_OF_FILE) throw InvalidGraphError("json_to_graph: Unexpected EOF in const_data array");
                            if (num_tok.type != TokType::NUMBER) {
                                throw InvalidGraphError("json_to_graph: const_data elements must be numbers");
                            }
                            if (!std::isfinite(num_tok.num_val) || std::abs(num_tok.num_val) > 3.4028235e38) {
                                throw InvalidGraphError("json_to_graph: const_data element is non-finite or overflows float32");
                            }
                            node.const_data.push_back(static_cast<float>(num_tok.num_val));
                        }
                    } else {
                        // Skip unknown field recursively (handles nested objects, arrays, and scalars)
                        Token val_tok = lexer.next();
                        lexer.skip_value(val_tok);
                    }
                }
                if (!node_closed) {
                    throw InvalidGraphError("json_to_graph: Unexpected EOF before closing '}' of node object");
                }
                if (!has_id || !has_op || !has_name || !has_shape || !has_dtype) {
                    throw InvalidGraphError("json_to_graph: node missing required property (id, op, name, shape, dtype are required)");
                }
                graph.add_node(std::move(node));
            }
        } else {
            // Skip other fields recursively
            Token val_tok = lexer.next();
            lexer.skip_value(val_tok);
        }
    }

    if (!closed) {
        throw InvalidGraphError("json_to_graph: Unexpected EOF before closing '}' of graph object");
    }

    if (!has_schema_major) {
        throw InvalidGraphError("json_to_graph: missing required root property 'schema_version_major'");
    }
    if (!has_inputs) {
        throw InvalidGraphError("json_to_graph: missing required root property 'inputs'");
    }
    if (!has_outputs) {
        throw InvalidGraphError("json_to_graph: missing required root property 'outputs'");
    }
    if (!has_nodes) {
        throw InvalidGraphError("json_to_graph: missing required root property 'nodes'");
    }

    Token trailing = lexer.next();
    if (trailing.type != TokType::END_OF_FILE) {
        throw InvalidGraphError("json_to_graph: Unexpected trailing non-whitespace characters after root JSON object");
    }

    graph.set_inputs(std::move(graph_inputs));
    graph.set_outputs(std::move(graph_outputs));

    return graph;
}

} // namespace forgetune
