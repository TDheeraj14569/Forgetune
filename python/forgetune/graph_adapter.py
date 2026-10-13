"""
Graph Adapter and Python IR representation.
Mirrors the C++ ForgeTune IR schema 1.0, and includes a Python reference interpreter.
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from .errors import InvalidGraphError, InvalidShapeError, UnsupportedOpError

class Node:
    def __init__(
        self,
        node_id: int,
        op: str,
        name: str = "",
        inputs: Optional[List[int]] = None,
        shape: Optional[List[int]] = None,
        dtype: str = "f32",
        const_data: Optional[List[float]] = None,
        deleted: bool = False
    ):
        self.id = node_id
        self.op = op.upper()
        self.name = name
        self.inputs = inputs or []
        self.shape = shape or []
        self.dtype = dtype
        self.const_data = const_data or []
        self.deleted = deleted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "name": self.name,
            "inputs": self.inputs,
            "shape": self.shape,
            "dtype": self.dtype,
            "const_data": self.const_data,
            "deleted": self.deleted
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        return cls(
            node_id=d["id"],
            op=d["op"],
            name=d.get("name", ""),
            inputs=d.get("inputs", []),
            shape=d.get("shape", []),
            dtype=d.get("dtype", "f32"),
            const_data=d.get("const_data", []),
            deleted=d.get("deleted", False)
        )

class Graph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.inputs: List[int] = []
        self.outputs: List[int] = []
        self._next_id = 0

    def add_node(self, node: Node) -> int:
        if node.id is None or node.id < 0:
            node.id = self._next_id
        if node.id in self.nodes:
            raise InvalidGraphError(f"Duplicate node ID: {node.id}")
        self.nodes[node.id] = node
        self._next_id = max(self._next_id, node.id + 1)
        return node.id

    def allocate_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def set_inputs(self, input_ids: List[int]) -> None:
        self.inputs = list(input_ids)

    def set_outputs(self, output_ids: List[int]) -> None:
        self.outputs = list(output_ids)

    def topological_order(self) -> List[int]:
        order = []
        visited = set()
        in_stack = set()

        def visit(nid: int):
            if nid in visited:
                return
            if nid in in_stack:
                raise InvalidGraphError(f"Cycle detected at node {nid}")
            node = self.nodes.get(nid)
            if not node or node.deleted:
                return

            in_stack.add(nid)
            for inp in node.inputs:
                visit(inp)
            in_stack.remove(nid)
            visited.add(nid)
            order.append(nid)

        for nid, node in list(self.nodes.items()):
            if not node.deleted:
                visit(nid)
        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version_major": 1,
            "schema_version_minor": 0,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [node.to_dict() for node in self.nodes.values()]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        g = cls()
        g.set_inputs(d.get("inputs", []))
        g.set_outputs(d.get("outputs", []))
        for nd in d.get("nodes", []):
            node = Node.from_dict(nd)
            g.nodes[node.id] = node
            g._next_id = max(g._next_id, node.id + 1)
        return g

    @classmethod
    def from_json(cls, json_str: str) -> "Graph":
        return cls.from_dict(json.loads(json_str))

    def evaluate(self, feed_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Pure-Python reference interpreter for differential correctness testing.
        """
        env: Dict[int, np.ndarray] = {}

        # Load inputs
        for in_id in self.inputs:
            if in_id not in feed_dict:
                raise InvalidGraphError(f"Missing input feed for node {in_id}")
            arr = np.asarray(feed_dict[in_id], dtype=np.float32)
            expected_shape = tuple(self.nodes[in_id].shape)
            if arr.shape != expected_shape:
                raise InvalidShapeError(
                    f"Input {in_id} shape mismatch: expected {expected_shape}, got {arr.shape}"
                )
            env[in_id] = arr

        for nid in self.topological_order():
            node = self.nodes[nid]
            if node.deleted:
                continue

            if node.op == "INPUT":
                continue
            elif node.op == "CONST":
                arr = np.array(node.const_data, dtype=np.float32).reshape(node.shape)
                env[nid] = arr
            elif node.op == "MATMUL":
                a = env[node.inputs[0]]
                b = env[node.inputs[1]]
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                    raise InvalidShapeError(f"Incompatible MATMUL shapes: {a.shape} @ {b.shape}")
                env[nid] = np.matmul(a, b)
            elif node.op == "ADD":
                a = env[node.inputs[0]]
                b = env[node.inputs[1]]
                # Strict broadcast contract:
                # 1. Exact shapes: A == B
                # 2. Rank-1 bias: A[M, N] + B[N]
                # 3. Rank-2 row: A[M, N] + B[1, N]
                is_valid = False
                if a.shape == b.shape:
                    is_valid = True
                elif a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
                    is_valid = True
                elif a.ndim == 2 and b.ndim == 2 and b.shape[0] == 1 and a.shape[1] == b.shape[1]:
                    is_valid = True
                if not is_valid:
                    raise InvalidShapeError(f"Unsupported ADD broadcast shapes: {a.shape} and {b.shape}")
                env[nid] = a + b
            elif node.op == "RELU":
                inp = env[node.inputs[0]]
                env[nid] = np.maximum(inp, 0.0, dtype=np.float32)
            elif node.op == "FUSED_MATMUL_ADD_RELU":
                x = env[node.inputs[0]]
                w = env[node.inputs[1]]
                b = env[node.inputs[2]]
                mm = np.matmul(x, w)
                added = mm + b
                env[nid] = np.maximum(added, 0.0, dtype=np.float32)
            elif node.op == "OUTPUT":
                env[nid] = env[node.inputs[0]]
            else:
                raise UnsupportedOpError(node.op)

        return {out_id: env[out_id] for out_id in self.outputs}
