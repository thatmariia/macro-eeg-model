from pydantic import BaseModel
from dataclasses import dataclass


class Node(BaseModel):
    name: str          # e.g. "parietal lobe"
    code: str          # e.g. "PL"
    is_relay: bool = False  # whether this node is a relay station

    def __hash__(self) -> int:
        # identity = (name, code)
        return hash((self.name, self.code))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (self.name, self.code) == (other.name, other.code)



@dataclass(slots=True)
class NodesCollection:
    nodes: list[Node]
    node_to_index: dict[Node, int]
    index_to_node: dict[int, Node]
    relay: Node | None = None

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}
        self.index_to_node = {idx: node for idx, node in enumerate(nodes)}
        self.relay = next((node for node in nodes if node.is_relay), None)

    def __post_init__(self) -> None:
        self._ensure_no_duplicates()
        self._ensure_at_most_one_relay()

    def _ensure_no_duplicates(self) -> None:
        seen: set[tuple[str, str]] = set()
        for node in self.nodes:
            key = (node.name, node.code)
            if key in seen:
                raise ValueError(f"duplicate node with name/code: {key}")
            seen.add(key)

    def _ensure_at_most_one_relay(self) -> None:
        relays = [node.code for node in self.nodes if node.is_relay]
        if len(relays) > 1:
            raise ValueError(
                f"at most one relay station allowed, found: {relays}"
            )
            