# -*- coding: utf-8 -*-
"""
Claim-evidence graph for Module 3.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from framework.memory import MemoryRetrievalResult
from framework.skills import EvidencePacket


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "data": self.data,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "data": self.data,
        }


@dataclass
class ClaimEvidenceGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "summary": self.summarize(),
        }

    def summarize(self) -> Dict[str, Any]:
        support_harmful = 0.0
        support_harmless = 0.0
        weaken_harmful = 0.0
        weaken_harmless = 0.0
        conflicts = 0

        for edge in self.edges:
            if edge.target == "Claim_Harmful" and edge.relation == "supports":
                support_harmful += edge.weight
            elif edge.target == "Claim_Harmless" and edge.relation == "supports":
                support_harmless += edge.weight
            elif edge.target == "Claim_Harmful" and edge.relation == "weakens":
                weaken_harmful += edge.weight
            elif edge.target == "Claim_Harmless" and edge.relation == "weakens":
                weaken_harmless += edge.weight
            elif edge.relation == "conflicts_with":
                conflicts += 1

        return {
            "support_harmful": support_harmful,
            "support_harmless": support_harmless,
            "weaken_harmful": weaken_harmful,
            "weaken_harmless": weaken_harmless,
            "num_conflicts": conflicts,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
        }

    def to_context_string(self) -> str:
        lines = ["## Claim-Evidence Graph"]
        for node in self.nodes:
            if node.node_type == "claim":
                lines.append(f"- Claim node: {node.node_id} ({node.label})")
        lines.append("\nEvidence relations:")
        for edge in self.edges:
            if edge.relation in {"supports", "weakens"}:
                lines.append(
                    f"- {edge.source} {edge.relation} {edge.target} "
                    f"(weight {edge.weight:.2f})"
                )
        summary = self.summarize()
        lines.append(
            "\nGraph summary: "
            f"support_harmful={summary['support_harmful']:.2f}, "
            f"support_harmless={summary['support_harmless']:.2f}, "
            f"weaken_harmful={summary['weaken_harmful']:.2f}, "
            f"weaken_harmless={summary['weaken_harmless']:.2f}, "
            f"conflicts={summary['num_conflicts']}"
        )
        return "\n".join(lines)


class EvidenceGraphBuilder:
    """Builds a two-claim graph from skill evidence packets."""

    def build(
        self,
        packets: List[EvidencePacket],
        memory_result: Optional[MemoryRetrievalResult] = None,
    ) -> ClaimEvidenceGraph:
        graph = ClaimEvidenceGraph()
        graph.nodes.extend([
            GraphNode("Claim_Harmful", "claim", "The meme is harmful"),
            GraphNode("Claim_Harmless", "claim", "The meme is harmless"),
        ])

        memory_ids = set(memory_result.get_memory_ids() if memory_result else [])
        evidence_claims: Dict[str, List[str]] = {}

        for idx, packet in enumerate(packets, 1):
            skill_id = f"skill_{packet.skill}"
            evidence_id = f"evidence_{idx}_{packet.skill}"
            graph.nodes.append(
                GraphNode(
                    skill_id,
                    "skill",
                    packet.skill,
                    {"confidence": packet.confidence},
                )
            )
            graph.nodes.append(
                GraphNode(
                    evidence_id,
                    "evidence",
                    packet.local_claim or packet.skill,
                    packet.to_dict(),
                )
            )
            graph.edges.append(
                GraphEdge(evidence_id, skill_id, "derived_from", packet.confidence)
            )

            evidence_claims[evidence_id] = []
            for claim in packet.supports:
                graph.edges.append(
                    GraphEdge(evidence_id, claim, "supports", packet.confidence)
                )
                evidence_claims[evidence_id].append(f"supports:{claim}")
            for claim in packet.weakens:
                graph.edges.append(
                    GraphEdge(evidence_id, claim, "weakens", packet.confidence)
                )
                evidence_claims[evidence_id].append(f"weakens:{claim}")

            for mem_id in packet.memory_support:
                if not memory_ids or mem_id in memory_ids:
                    graph.nodes.append(GraphNode(mem_id, "memory", mem_id))
                    graph.edges.append(
                        GraphEdge(evidence_id, mem_id, "supported_by_memory", 1.0)
                    )

            if packet.uncertainty:
                uncertainty_id = f"uncertainty_{idx}_{packet.skill}"
                graph.nodes.append(
                    GraphNode(
                        uncertainty_id,
                        "uncertainty",
                        "; ".join(packet.uncertainty),
                    )
                )
                graph.edges.append(
                    GraphEdge(uncertainty_id, evidence_id, "qualifies", 1.0)
                )

        evidence_ids = list(evidence_claims)
        for i, left in enumerate(evidence_ids):
            for right in evidence_ids[i + 1:]:
                if _relations_conflict(evidence_claims[left], evidence_claims[right]):
                    graph.edges.append(GraphEdge(left, right, "conflicts_with", 1.0))
                    graph.edges.append(GraphEdge(right, left, "conflicts_with", 1.0))

        return graph


def _relations_conflict(left: List[str], right: List[str]) -> bool:
    left_set = set(left)
    right_set = set(right)
    return bool(
        ("supports:Claim_Harmful" in left_set and "supports:Claim_Harmless" in right_set)
        or ("supports:Claim_Harmless" in left_set and "supports:Claim_Harmful" in right_set)
        or ("supports:Claim_Harmful" in left_set and "weakens:Claim_Harmful" in right_set)
        or ("weakens:Claim_Harmful" in left_set and "supports:Claim_Harmful" in right_set)
        or ("supports:Claim_Harmless" in left_set and "weakens:Claim_Harmless" in right_set)
        or ("weakens:Claim_Harmless" in left_set and "supports:Claim_Harmless" in right_set)
    )

