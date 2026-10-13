
from __future__ import annotations

import itertools
import json
from pathlib import Path

import networkx as nx


def _node_key(name: str, entity_type: str) -> str:
    return f"{entity_type}:{name.lower()}"


class CooccurrenceGraphBuilder:
    def __init__(self, graph: nx.Graph | None = None):
        self.graph = graph if graph is not None else nx.Graph()

    def add_chunk_entities(self, chunk_id: int, entities: list[tuple[str, str]]) -> None:

        keys: list[str] = []
        for name, entity_type in entities:
            key = _node_key(name, entity_type)
            keys.append(key)
            if key not in self.graph:
                self.graph.add_node(key, name=name, type=entity_type, chunk_ids=[])
            if chunk_id not in self.graph.nodes[key]["chunk_ids"]:
                self.graph.nodes[key]["chunk_ids"].append(chunk_id)

        for key_a, key_b in itertools.combinations(sorted(set(keys)), 2):
            if self.graph.has_edge(key_a, key_b):
                self.graph[key_a][key_b]["weight"] += 1
            else:
                self.graph.add_edge(key_a, key_b, weight=1)

    def remove_chunks(self, chunk_ids: set[int]) -> None:
        # Nodes are shared across users/documents (see graph_expansion.py), so a node
        # is dropped only once no chunk_ids reference it at all, not on first mention removed.
        empty_nodes = []
        for key, data in self.graph.nodes(data=True):
            remaining = [cid for cid in data.get("chunk_ids", []) if cid not in chunk_ids]
            if remaining != data["chunk_ids"]:
                data["chunk_ids"] = remaining
            if not remaining:
                empty_nodes.append(key)
        self.graph.remove_nodes_from(empty_nodes)

    def neighbors(self, name: str, entity_type: str) -> list[str]:
        key = _node_key(name, entity_type)
        if key not in self.graph:
            return []
        return list(self.graph.neighbors(key))

    @staticmethod
    def node_key(name: str, entity_type: str) -> str:
        return _node_key(name, entity_type)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.adjacency_data(self.graph)
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> CooccurrenceGraphBuilder:
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(nx.adjacency_graph(data))