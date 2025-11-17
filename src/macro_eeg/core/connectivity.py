from __future__ import annotations
from xxlimited import new
from pydantic import BaseModel, model_validator, Field
from .nodes import Node, NodesCollection
from typing import TypeVar, Callable, Iterable, Type
from dataclasses import dataclass
import numpy as np


class Edge(BaseModel):
    src: Node   # full node
    dst: Node   # full node

    def with_reversed(self):
        """
        Return the same object but with src/dst swapped.
        """
        return self, Edge(src=self.dst, dst=self.src)


class _NoSelfLoopEdge(BaseModel):
    edge: Edge

    @model_validator(mode="after")
    def _no_self_loop(self):
        if self.edge.src == self.edge.dst:
            raise ValueError(
                f"self-loop not allowed: {self.edge.src.name} ({self.edge.src.code}) "
                f"-> {self.edge.dst.name} ({self.edge.dst.code})"
            )
        return self

    def with_reversed(self):
        """
        Return the same object but with src/dst swapped.
        Works for subclasses because we use model_copy().
        """
        return self, self.model_copy(
            update={
                "edge": Edge(src=self.edge.dst, dst=self.edge.src),
            }
        )


class EdgeConnectivity(_NoSelfLoopEdge):
    weight: float = Field(gt=0)
    
    @model_validator(mode="after")
    def _check_no_relay_connection(self):
        if self.edge.src.is_relay or self.edge.dst.is_relay:
            raise ValueError(
                f"connectivity edges cannot involve relay stations: "
                f"{self.edge.src.code} -> {self.edge.dst.code}"
            )
        return self


class EdgeConnectivityCoef(_NoSelfLoopEdge):
    coef: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_no_relay_connection(self):
        if self.edge.src.is_relay or self.edge.dst.is_relay:
            raise ValueError(
                f"connectivity coefficient edges cannot involve relay stations: "
                f"{self.edge.src.code} -> {self.edge.dst.code}"
            )
        return self


class EdgeDistance(_NoSelfLoopEdge):
    distance: float = Field(gt=0)


@dataclass(slots=True)
class EdgesCollection:

    edges: list[_NoSelfLoopEdge]
    matrix: np.ndarray
    nodes_collection: NodesCollection

    edge_type: Type[_NoSelfLoopEdge]
    value_attr: str

    _attr_map = {
        EdgeConnectivity: "weight",
        EdgeConnectivityCoef: "coef",
        EdgeDistance: "distance",
    }

    # ----------------- core validation -----------------

    def __post_init__(self) -> None:
        self._validate_edge_type()
        self._validate_matrix_shape()
        self._validate_value_attr()
        self._ensure_edge_nodes_in_collection()
        self._ensure_no_duplicate_edges()

    def _validate_edge_type(self) -> None:
        for edge in self.edges:
            if not isinstance(edge, self.edge_type):
                raise ValueError(
                    f"all edges must be of type {self.edge_type}, found {type(edge)}"
                )

    def _validate_matrix_shape(self) -> None:
        n_nodes = len(self.nodes_collection.nodes)

        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square")

        if self.matrix.shape[0] > n_nodes:
            raise ValueError(
                f"matrix shape {self.matrix.shape} incompatible with "
                f"number of nodes {n_nodes} in nodes_collection"
            )

    def _validate_value_attr(self) -> None:
        for edge in self.edges:
            val = getattr(edge, self.value_attr, None)
            if val is None:
                raise ValueError(
                    f"edge missing expected value attribute '{self.value_attr}'"
                )

    def _ensure_edge_nodes_in_collection(self) -> None:
        for edge in self.edges:
            if edge.edge.src not in self.nodes_collection.nodes:
                raise ValueError(
                    f"edge source node {edge.edge.src.code} not in nodes_collection"
                    f"Edge: {edge.edge.src.code} -> {edge.edge.dst.code}"
                )
            if edge.edge.dst not in self.nodes_collection.nodes:
                raise ValueError(
                    f"edge destination node {edge.edge.dst.code} not in nodes_collection"
                    f"Edge: {edge.edge.src.code} -> {edge.edge.dst.code}"
                )

    def _ensure_no_duplicate_edges(self) -> None:
        seen: set[tuple[Node, Node]] = set()
        for edge in self.edges:
            key = (edge.edge.src, edge.edge.dst)
            if key in seen:
                raise ValueError(f"duplicate edge {edge.edge.src.code} -> {edge.edge.dst.code}")
            seen.add(key)

    # ----------------- constructors -----------------

    @classmethod
    def from_edges(
        cls,
        edges: list[_NoSelfLoopEdge],
        *,
        nodes_collection: NodesCollection,
        default_value: float = 0.0,
    ) -> EdgesCollection:
        """
        Build an EdgesCollection from a list of edges.
        """

        edge_type = type(edges[0])
        for edge in edges:
            if type(edge) is not edge_type:
                raise ValueError(
                    "EdgesCollection must be homogeneous (all edges same class)"
                )

        value_attr = cls._attr_map.get(edge_type)
        if value_attr is None:
            raise ValueError(f"unsupported edge type: {edge_type}")

        n_nodes = len(nodes_collection.nodes)
        mat = np.ones((n_nodes, n_nodes), dtype=float) * default_value
        for edge in edges:
            i = nodes_collection.node_to_index.get(edge.edge.src, None)
            j = nodes_collection.node_to_index.get(edge.edge.dst, None)
            if i is None or j is None:
                raise ValueError(
                    f"edge nodes {edge.edge.src.code} or {edge.edge.dst.code} not in nodes_collection"
                )
            mat[i, j] = getattr(edge, value_attr)

        return cls(
            edges=edges,
            matrix=mat,
            nodes_collection=nodes_collection,
            edge_type=edge_type,
            value_attr=value_attr,
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        nodes_collection: NodesCollection,
        edge_type: Type[_NoSelfLoopEdge],
        ignore_zeros: bool = True,
    ) -> EdgesCollection:
        """
        Build an EdgesCollection from a matrix, reconstructing edge objects of
        the given `edge_type` using the appropriate value attribute.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")

        if matrix.shape[0] > len(nodes_collection.nodes):
            raise ValueError(
                "matrix size larger than number of nodes in nodes_collection"
            )

        value_attr = cls._attr_map.get(edge_type)
        if value_attr is None:
            raise ValueError(f"unsupported edge type: {edge_type}")

        edges: list[_NoSelfLoopEdge] = []
        n_nodes = len(nodes_collection.nodes)

        for i in range(n_nodes):
            for j in range(n_nodes):
                val = float(matrix[i, j])
                if val == 0.0 and ignore_zeros:
                    continue
                src = nodes_collection.index_to_node.get(i, None)
                dst = nodes_collection.index_to_node.get(j, None)
                if src is None or dst is None:
                    raise ValueError(
                        f"node index {i} or {j} not in nodes_collection"
                    )
                edge = Edge(src=src, dst=dst)
                # instantiate the right type; Pydantic will run its validators
                edge_obj = edge_type(edge=edge, **{value_attr: val})
                edges.append(edge_obj)

        return cls(
            edges=edges,
            matrix=matrix,
            nodes_collection=nodes_collection,
            edge_type=edge_type,
            value_attr=value_attr,
        )

    # ----------------- "update" helpers -----------------

    def update_with_edges(self, edges: list[_NoSelfLoopEdge]) -> EdgesCollection:
        """
        Return a NEW EdgesCollection with updated edges and recomputed matrix.
        """
        return type(self).from_edges(
            edges=edges,
            nodes_collection=self.nodes_collection,
        )

    def update_with_matrix(self, matrix: np.ndarray) -> EdgesCollection:
        """
        Return a NEW EdgesCollection with updated edges and recomputed matrix.
        """
        return type(self).from_matrix(
            matrix=matrix,
            nodes_collection=self.nodes_collection,
            edge_type=self.edge_type,
        )

    # ----------------- utility functions -----------------

    def _get_direct_edge_value(self, src: Node, dst: Node) -> float:
        """
        Return the value attribute for the direct edge src -> dst.

        Raises ValueError if no such edge exists.
        """
        for edge in self.edges:
            if edge.edge.src == src and edge.edge.dst == dst:
                return getattr(edge, self.value_attr)
        raise ValueError(f"no edge found from {src.code} to {dst.code}")

    def get_edge_value_from_nodes(
        self,
        src: Node,
        dst: Node,
        *,
        relayed: bool = True,
    ) -> float | tuple[float, float]:
        """
        Get the value attribute of the edge from src to dst, or None if no such edge.
        """
        if not relayed:
            return self._get_direct_edge_value(src, dst)

        relay = self.nodes_collection.relay
        if relay is None:
            # TODO: warning
            return self._get_direct_edge_value(src, dst)
            # raise ValueError("no relay station defined in nodes_collection")

        val1 = self._get_direct_edge_value(src, relay)
        val2 = self._get_direct_edge_value(relay, dst)

        return val1, val2