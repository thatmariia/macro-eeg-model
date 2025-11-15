# yourpkg/models/stimuli.py
from pydantic import BaseModel, model_validator, Field, ConfigDict, field_serializer, field_validator
from .nodes import Node, NodesCollection
from .connectivity import EdgeConnectivityCoef, EdgesCollection
from .types import StimulusCallable, TimeCallable
import numpy as np


class StimulusNodeTarget(BaseModel):
    # "which node" and "how much of the stimulus goes there"
    node: Node
    coef: float = 1.0

    @model_validator(mode="after")
    def _ensure_no_relay_node(self):
        if self.node.is_relay:
            raise ValueError(f"stimulus target node '{self.node.code}' cannot be a relay station")
        return self


class Stimulus(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    node_targets: list[StimulusNodeTarget]
    onset_ms: int | TimeCallable
    duration_ms: int | TimeCallable
    edge_coeffs: EdgesCollection | None = None
    stimulus_fn: StimulusCallable | None

    @field_validator("onset_ms", "duration_ms")
    def validate_time_field(cls, v):
        if isinstance(v, int):
            if v < 0:
                raise ValueError("time values must be non-negative")
        return v

    @field_serializer("edge_coeffs")
    def serialize_edges_collection(self, v: EdgesCollection):
        return {
            "matrix": v.matrix.tolist(),  # np.ndarray → list[list[float]]
            "nodes": [
                {"name": n.name, "code": n.code} for n in v.nodes_collection.nodes
            ],
            "edges": [
                {
                    "src": e.edge.src.code,
                    "dst": e.edge.dst.code,
                    v.value_attr: getattr(e, v.value_attr),
                }
                for e in v.edges
            ],
            "edge_type": v.edge_type.__name__,
        }

    @model_validator(mode="after")
    def _ensure_no_duplicate_node_targets(self):
        seen: set[Node] = set()
        for target in self.node_targets:
            node = target.node
            if node in seen:
                raise ValueError(f"duplicate node target for node {node.code} in stimulus '{self.name}'")
            seen.add(node)
        return self

    @model_validator(mode="after")
    def _validate_edge_collection_type(self):
        if self.edge_coeffs is not None:
            if self.edge_coeffs.edge_type is not EdgeConnectivityCoef:
                raise ValueError(
                    f"stimulus edge_coeffs must be of type EdgeConnectivityCoef, "
                    f"found {self.edge_coeffs.edge_type}"
                )
        return self

    @model_validator(mode="after")
    def _ensure_node_targets_in_edge_nodes_collection(self):
        if self.edge_coeffs is None:
            return self
        edge_nodes = self.edge_coeffs.nodes_collection.nodes
        for target in self.node_targets:
            if target.node not in edge_nodes:
                raise ValueError(
                    f"stimulus target node '{target.node.code}' not in edge_coeffs nodes collection"
                )
        return self

    def is_active_at(self, t_ms: int) -> bool:
        if isinstance(self.onset_ms, int) and isinstance(self.duration_ms, int):
            return self.onset_ms <= t_ms < self.onset_ms + self.duration_ms

        raise ValueError("cannot determine activity at time for stimuli with dynamic onset/duration")

    def target_coefs(self, nodes_collection: NodesCollection) -> np.ndarray:
        """
        Get array of target coefficients for all nodes in the given nodes collection.

        Parameters
        ----------
        nodes_collection : NodesCollection
            Collection of nodes to map the stimulus targets onto.

        Returns
        -------
        (N,) array
            Array of target coefficients, where N is the number of nodes in the collection.
            Coefficients are zero for nodes not targeted by the stimulus.
        """
        target_array = np.zeros(len(nodes_collection.nodes), dtype=float)

        for target in self.node_targets:
            if target.node in nodes_collection.nodes:
                idx = nodes_collection.node_to_index[target.node]
                target_array[idx] = target.coef

        return target_array