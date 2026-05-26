# -*- coding: utf-8 -*-
"""
Skill-memory framework for zero-shot harmful meme detection.

Primary flow:
1. Build label-free case / insight / risk memory.
2. Extract query signature and retrieve memory.
3. Plan and execute reasoning skills.
4. Build a harmful / harmless claim-evidence graph.
5. Judge from the graph and optionally update memory.
"""

from framework.config import (
    AVAILABLE_MODELS,
    DATASET_CONFIGS,
    DEFAULT_FRAMEWORK_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_PATH_CONFIG,
    FrameworkConfig,
    ModelConfig,
    PathConfig,
)
from framework.direct_baseline import DirectBaselineResult, DirectMemeBaseline
from framework.evidence_graph import (
    ClaimEvidenceGraph,
    EvidenceGraphBuilder,
    GraphEdge,
    GraphNode,
)
from framework.judge import (
    GraphJudge,
    GraphJudgmentResult,
    MemoryCurator,
)
from framework.memory import (
    CaseMemoryItem,
    InsightMemoryItem,
    MemoryRetrievalResult,
    MemoryUpdateDecision,
    QuerySignature,
    RiskMemoryItem,
    ScoredMemory,
    SkillMemoryStore,
)
from framework.pipeline import (
    AblationPipeline,
    MemeDetectionPipeline,
    PipelineResult,
)
from framework.planner import (
    SkillPlan,
    SkillPlanner,
)
from framework.signature import QuerySignatureExtractor
from framework.skills import (
    DEFAULT_SKILL_CARDS,
    EvidencePacket,
    ReasoningSkillExecutor,
    SkillCard,
    SkillInducer,
    SkillRepository,
    SkillType,
    SkillValidator,
)

__version__ = "2.0.0"

__all__ = [
    "AVAILABLE_MODELS",
    "DATASET_CONFIGS",
    "DEFAULT_FRAMEWORK_CONFIG",
    "DEFAULT_MODEL",
    "DEFAULT_PATH_CONFIG",
    "FrameworkConfig",
    "ModelConfig",
    "PathConfig",
    "DirectBaselineResult",
    "DirectMemeBaseline",
    "CaseMemoryItem",
    "InsightMemoryItem",
    "MemoryRetrievalResult",
    "MemoryUpdateDecision",
    "QuerySignature",
    "RiskMemoryItem",
    "ScoredMemory",
    "SkillMemoryStore",
    "DEFAULT_SKILL_CARDS",
    "EvidencePacket",
    "ReasoningSkillExecutor",
    "SkillCard",
    "SkillInducer",
    "SkillRepository",
    "SkillType",
    "SkillValidator",
    "SkillPlan",
    "SkillPlanner",
    "QuerySignatureExtractor",
    "GraphNode",
    "GraphEdge",
    "ClaimEvidenceGraph",
    "EvidenceGraphBuilder",
    "GraphJudge",
    "GraphJudgmentResult",
    "MemoryCurator",
    "MemeDetectionPipeline",
    "AblationPipeline",
    "PipelineResult",
]
