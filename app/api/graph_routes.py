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
