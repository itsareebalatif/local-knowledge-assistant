"""Graph inspection endpoints (SRS 1.4: "fetch graph relationships").

GET /graph returns the full co-occurrence graph as adjacency data (the same
shape CooccurrenceGraphBuilder.save()/load() already use — reusing it here
rather than inventing a second JSON shape for the same graph).

GET /graph/neighbors is the more commonly useful call: 1-hop neighbors of
one named entity, with the co-occurrence weight connecting them — the same
traversal app.graph.graph_expansion uses internally during retrieval, now
inspectable directly for debugging "why did the graph pull in chunk X".

Both require login but the graph structure itself is NOT filtered per user:
entities and co-occurrence edges are shared structure (what concepts relate
to what), not document content — the actual chunk text behind a node stays
protected by chunk_routes.py's ownership check and retrieval's user_id
filters. This is a deliberate scope line, not an oversight.
"""

from __future__ import annotations

import networkx as nx
from fastapi import APIRouter, Depends, Query

from app.api.schemas import GraphNeighbor, GraphNeighborsResponse
from app.dependencies import get_current_user, get_graph_builder
from app.graph.graph_builder import CooccurrenceGraphBuilder
from app.models import User

router = APIRouter()


@router.get("/graph", tags=["graph"], summary="Export the full co-occurrence graph")
def get_graph(
    current_user: User = Depends(get_current_user),
    graph_builder: CooccurrenceGraphBuilder = Depends(get_graph_builder),
) -> dict:
    return nx.adjacency_data(graph_builder.graph)


@router.get(
    "/graph/neighbors",
    response_model=GraphNeighborsResponse,
    tags=["graph"],
    summary="1-hop neighbors of one entity, with co-occurrence weight",
)
def get_graph_neighbors(
    name: str = Query(..., description="Entity name, e.g. 'Ollama'"),
    type: str = Query(..., description="Entity type: Person, Organization, Location, or Concept"),
    current_user: User = Depends(get_current_user),
    graph_builder: CooccurrenceGraphBuilder = Depends(get_graph_builder),
) -> GraphNeighborsResponse:
    key = CooccurrenceGraphBuilder.node_key(name, type)
    entity_label = f"{type}:{name}"

    if key not in graph_builder.graph:
        return GraphNeighborsResponse(entity=entity_label, neighbors=[])

    neighbors = []
    for neighbor_key in graph_builder.graph.neighbors(key):
        node = graph_builder.graph.nodes[neighbor_key]
        weight = graph_builder.graph[key][neighbor_key].get("weight", 1)
        neighbors.append(GraphNeighbor(name=node.get("name", neighbor_key), type=node.get("type", ""), weight=weight))

    neighbors.sort(key=lambda n: n.weight, reverse=True)
    return GraphNeighborsResponse(entity=entity_label, neighbors=neighbors)
