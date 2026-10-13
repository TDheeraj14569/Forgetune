#ifndef FORGETUNE_JSON_IO_HPP
#define FORGETUNE_JSON_IO_HPP

#include "forgetune/ir.hpp"
#include <string>

namespace forgetune {

// Serialize an IRGraph to a JSON string
std::string graph_to_json(const IRGraph& graph);

// Deserialize an IRGraph from a JSON string
IRGraph json_to_graph(const std::string& json_str);

} // namespace forgetune

#endif // FORGETUNE_JSON_IO_HPP
