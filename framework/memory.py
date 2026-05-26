# -*- coding: utf-8 -*-
"""
Label-free skill-memory construction and retrieval.

This module implements Module 1 from ``方法模块.md``:

1. Case Memory: structured context for individual unlabeled memes.
2. Insight Memory: reusable judgment principles induced from related cases.
3. Risk Memory: recurring ambiguity, disagreement, and evidence-conflict modes.

The implementation is deliberately label-free at inference time.  Dataset labels
may exist in the raw files for evaluation, but they are not exposed through
retrieved memory contexts used by the planner or judge.
"""
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    OpenAI = None
from tqdm import tqdm

from framework.config import (
    DATASET_CONFIGS,
    DEFAULT_FRAMEWORK_CONFIG,
    DEFAULT_MODEL,
    canonical_data_name,
)
from framework.llm_utils import LLMCaller, clamp_float, extract_json_object


STOPWORDS = {
    "about", "again", "after", "against", "also", "because", "being",
    "between", "could", "every", "from", "have", "into", "like", "more",
    "most", "only", "over", "should", "some", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "under", "very", "what", "when", "where", "which", "while", "with",
    "would", "your", "youre",
}


HARM_KEYWORDS = {
    "protected_group": [
        "black", "white", "asian", "jew", "jewish", "muslim", "christian",
        "gay", "lesbian", "trans", "immigrant", "mexican", "race", "racist",
        "religion", "gender", "women", "woman", "men", "disabled",
    ],
    "misogyny": [
        "women", "woman", "girl", "girls", "feminist", "feminism", "wife",
        "kitchen", "slut", "bitch", "lesbian",
    ],
    "violence": [
        "beat", "kill", "shoot", "hang", "attack", "punch", "bomb", "die",
        "death", "dead", "murder",
    ],
    "political": [
        "trump", "biden", "obama", "clinton", "president", "government",
        "politics", "election", "vote", "liberal", "conservative",
    ],
    "historical_harm": [
        "hitler", "nazi", "holocaust", "slavery", "war", "genocide",
        "colonial", "terrorist", "terrorism",
    ],
    "health_or_disaster": [
        "covid", "virus", "pandemic", "vaccine", "cancer", "aids",
        "earthquake", "flood", "disaster",
    ],
    "sexualization": [
        "sex", "sexual", "handjob", "blowjob", "porn", "nude", "boobs",
        "ass", "rape",
    ],
}


STATIC_RISKS = [
    {
        "risk_type": "target_boundary_confusion",
        "description": (
            "The model may confuse attacks on protected groups with satire, "
            "self-deprecation, or criticism of a public figure."
        ),
        "triggers": ["protected group mention", "public figure", "ambiguous target"],
        "recommended_skills": [
            "target_boundary",
            "protected_group_attribution",
            "political_criticism_boundary",
        ],
    },
    {
        "risk_type": "quoted_criticism_confusion",
        "description": (
            "Harmful language may be quoted in order to criticize it rather than "
            "promote it."
        ),
        "triggers": ["quotation", "mocking hate", "counter speech", "ironic setup"],
        "recommended_skills": ["counter_speech_quoted_criticism", "target_boundary"],
    },
    {
        "risk_type": "surface_incongruity_overweighting",
        "description": (
            "Image-text mismatch or sarcasm can be benign humor; it should not be "
            "treated as harmful without a harm mechanism or target."
        ),
        "triggers": ["image text mismatch", "sarcasm", "benign joke"],
        "recommended_skills": ["image_text_incongruity", "target_boundary"],
    },
    {
        "risk_type": "political_satire_boundary",
        "description": (
            "Political criticism and public-figure mockery are not automatically "
            "harmful, but can become harmful when they normalize violence, "
            "dehumanization, or dangerous misinformation."
        ),
        "triggers": ["political figure", "government", "crisis", "disease"],
        "recommended_skills": [
            "political_criticism_boundary",
            "historical_harm_trivialization",
            "target_boundary",
        ],
    },
    {
        "risk_type": "misogyny_vs_relationship_humor",
        "description": (
            "Relationship or gender jokes can be harmless, but recurring framing "
            "that objectifies, demeans, or stereotypes women should be treated as "
            "a misogyny risk."
        ),
        "triggers": ["women", "wife", "girlfriend", "sexualization"],
        "recommended_skills": ["misogynistic_framing", "protected_group_attribution"],
    },
]

MEMORY_SCHEMA_VERSION = "skill_memory_label_free_v3"


@dataclass
class QuerySignature:
    """Structured problem signature extracted from the current meme."""

    visual_summary: str = ""
    ocr_text: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    tone: str = "unknown"
    image_text_relation: str = "unknown"
    possible_harm_mechanisms: List[str] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_retrieval_text(self, caption: str = "") -> str:
        parts = [
            caption,
            self.visual_summary,
            " ".join(self.ocr_text),
            " ".join(self.entities),
            " ".join(self.target_candidates),
            self.tone,
            self.image_text_relation,
            " ".join(self.possible_harm_mechanisms),
            " ".join(self.uncertainty),
        ]
        return " ".join(p for p in parts if p).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual_summary": self.visual_summary,
            "ocr_text": self.ocr_text,
            "entities": self.entities,
            "target_candidates": self.target_candidates,
            "tone": self.tone,
            "image_text_relation": self.image_text_relation,
            "possible_harm_mechanisms": self.possible_harm_mechanisms,
            "uncertainty": self.uncertainty,
            "raw": self.raw,
        }


@dataclass
class CaseMemoryItem:
    """Structured context for one reference meme."""

    memory_id: str
    source_id: str
    image_filename: Optional[str]
    text: str
    visual_summary: str = ""
    ocr_text: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    target_candidates: List[str] = field(default_factory=list)
    possible_harm_mechanisms: List[str] = field(default_factory=list)
    supports_harmful: List[str] = field(default_factory=list)
    supports_harmless: List[str] = field(default_factory=list)
    uncertainty: List[str] = field(default_factory=list)
    prior_explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def surface_text(self) -> str:
        parts = [
            self.text,
            self.visual_summary,
            " ".join(self.ocr_text),
            " ".join(self.entities[:8]),
        ]
        return " ".join(p for p in parts if p).strip()

    def retrieval_text(self) -> str:
        parts = [
            self.text,
            self.visual_summary,
            " ".join(self.ocr_text),
            " ".join(self.entities),
            " ".join(self.target_candidates),
            " ".join(self.possible_harm_mechanisms),
            " ".join(self.supports_harmful),
            " ".join(self.supports_harmless),
            " ".join(self.uncertainty),
            self.prior_explanation or "",
        ]
        return " ".join(p for p in parts if p).strip()

    def to_context_string(self, include_explanation: bool = True) -> str:
        lines = [
            f"Case {self.memory_id}",
            f"Text: \"{self.text}\"",
        ]
        if self.visual_summary:
            lines.append(f"Visual summary: {self.visual_summary}")
        if self.entities:
            lines.append(f"Entities: {', '.join(self.entities)}")
        if self.target_candidates:
            lines.append(f"Target candidates: {', '.join(self.target_candidates)}")
        if self.possible_harm_mechanisms:
            lines.append(
                "Possible harm mechanisms: "
                + ", ".join(self.possible_harm_mechanisms)
            )
        if self.supports_harmful:
            lines.append("Evidence for harmful: " + "; ".join(self.supports_harmful))
        if self.supports_harmless:
            lines.append("Evidence for harmless: " + "; ".join(self.supports_harmless))
        if self.uncertainty:
            lines.append("Uncertainty: " + "; ".join(self.uncertainty))
        if include_explanation and self.prior_explanation:
            lines.append(f"Reference explanation: {self.prior_explanation}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "source_id": self.source_id,
            "image_filename": self.image_filename,
            "text": self.text,
            "visual_summary": self.visual_summary,
            "ocr_text": self.ocr_text,
            "entities": self.entities,
            "target_candidates": self.target_candidates,
            "possible_harm_mechanisms": self.possible_harm_mechanisms,
            "supports_harmful": self.supports_harmful,
            "supports_harmless": self.supports_harmless,
            "uncertainty": self.uncertainty,
            "prior_explanation": self.prior_explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseMemoryItem":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class InsightMemoryItem:
    """Generalized judgment experience induced from related cases."""

    memory_id: str
    title: str
    pattern: str
    applicable_when: List[str]
    judgment_guidance: str
    related_case_ids: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    reliability: float = 0.6

    def retrieval_text(self) -> str:
        return " ".join([
            self.title,
            self.pattern,
            " ".join(self.applicable_when),
            self.judgment_guidance,
            " ".join(self.risk_notes),
        ]).strip()

    def to_context_string(self) -> str:
        lines = [
            f"Insight {self.memory_id}: {self.title}",
            f"Pattern: {self.pattern}",
            "Applicable when: " + "; ".join(self.applicable_when),
            f"Guidance: {self.judgment_guidance}",
            f"Reliability: {self.reliability:.2f}",
        ]
        if self.risk_notes:
            lines.append("Risk notes: " + "; ".join(self.risk_notes))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "title": self.title,
            "pattern": self.pattern,
            "applicable_when": self.applicable_when,
            "judgment_guidance": self.judgment_guidance,
            "related_case_ids": self.related_case_ids,
            "risk_notes": self.risk_notes,
            "reliability": self.reliability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InsightMemoryItem":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class RiskMemoryItem:
    """Known ambiguity or failure mode."""

    memory_id: str
    risk_type: str
    description: str
    triggers: List[str]
    recommended_skills: List[str]
    failure_mode: str = ""
    reliability: float = 0.6

    def retrieval_text(self) -> str:
        return " ".join([
            self.risk_type,
            self.description,
            " ".join(self.triggers),
            " ".join(self.recommended_skills),
            self.failure_mode,
        ]).strip()

    def to_context_string(self) -> str:
        lines = [
            f"Risk {self.memory_id}: {self.risk_type}",
            f"Description: {self.description}",
            "Triggers: " + "; ".join(self.triggers),
            "Recommended skills: " + ", ".join(self.recommended_skills),
            f"Reliability: {self.reliability:.2f}",
        ]
        if self.failure_mode:
            lines.append(f"Failure mode: {self.failure_mode}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "risk_type": self.risk_type,
            "description": self.description,
            "triggers": self.triggers,
            "recommended_skills": self.recommended_skills,
            "failure_mode": self.failure_mode,
            "reliability": self.reliability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskMemoryItem":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class ScoredMemory:
    item: Any
    score: float
    source: str = "logic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "source": self.source,
            "item": self.item.to_dict(),
        }


@dataclass
class MemoryRetrievalResult:
    """Retrieved memory used by the planner, skills, and judge."""

    query_text: str
    case_memories: List[ScoredMemory] = field(default_factory=list)
    logic_case_memories: List[ScoredMemory] = field(default_factory=list)
    surface_case_memories: List[ScoredMemory] = field(default_factory=list)
    insight_memories: List[ScoredMemory] = field(default_factory=list)
    risk_memories: List[ScoredMemory] = field(default_factory=list)
    retrieval_diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_examples(self) -> List[CaseMemoryItem]:
        """Compatibility alias for older router/adjudicator patterns."""
        return [scored.item for scored in self.case_memories]

    def get_context_string(
        self,
        top_k_cases: Optional[int] = None,
        top_k_insights: Optional[int] = None,
        top_k_risks: Optional[int] = None,
    ) -> str:
        case_rows = self.case_memories[:top_k_cases] if top_k_cases else self.case_memories
        insight_rows = (
            self.insight_memories[:top_k_insights]
            if top_k_insights else self.insight_memories
        )
        risk_rows = self.risk_memories[:top_k_risks] if top_k_risks else self.risk_memories

        parts = []
        if case_rows:
            parts.append("## Case Memory")
            for i, row in enumerate(case_rows, 1):
                parts.append(
                    f"\n### Retrieved Case {i} "
                    f"(score {row.score:.3f}, source {row.source})\n"
                    + row.item.to_context_string()
                )
        if insight_rows:
            parts.append("\n## Insight Memory")
            for i, row in enumerate(insight_rows, 1):
                parts.append(
                    f"\n### Retrieved Insight {i} (score {row.score:.3f})\n"
                    + row.item.to_context_string()
                )
        if risk_rows:
            parts.append("\n## Risk Memory")
            for i, row in enumerate(risk_rows, 1):
                parts.append(
                    f"\n### Retrieved Risk {i} (score {row.score:.3f})\n"
                    + row.item.to_context_string()
                )
        return "\n".join(parts) if parts else "No relevant memory retrieved."

    def get_memory_ids(self) -> List[str]:
        ids = []
        for rows in [self.case_memories, self.insight_memories, self.risk_memories]:
            ids.extend(row.item.memory_id for row in rows)
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "case_memories": [row.to_dict() for row in self.case_memories],
            "logic_case_memories": [row.to_dict() for row in self.logic_case_memories],
            "surface_case_memories": [row.to_dict() for row in self.surface_case_memories],
            "insight_memories": [row.to_dict() for row in self.insight_memories],
            "risk_memories": [row.to_dict() for row in self.risk_memories],
            "retrieval_diagnostics": self.retrieval_diagnostics,
        }


@dataclass
class MemoryUpdateDecision:
    """Label-free update proposal produced after graph judgment."""

    write_case: bool = False
    write_insight: bool = False
    write_risk: bool = False
    case_item: Optional[CaseMemoryItem] = None
    insight_item: Optional[InsightMemoryItem] = None
    risk_item: Optional[RiskMemoryItem] = None
    skill_reliability_updates: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "write_case": self.write_case,
            "write_insight": self.write_insight,
            "write_risk": self.write_risk,
            "case_item": self.case_item.to_dict() if self.case_item else None,
            "insight_item": self.insight_item.to_dict() if self.insight_item else None,
            "risk_item": self.risk_item.to_dict() if self.risk_item else None,
            "skill_reliability_updates": self.skill_reliability_updates,
            "rationale": self.rationale,
        }


class SkillMemoryStore:
    """Persistent store for case, insight, and risk memory."""

    def __init__(
        self,
        dataset_name: str,
        config: Optional[Any] = None,
        memory_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        self.data_dir_name = canonical_data_name(dataset_name)
        self.memory_dir = memory_dir or os.path.join("data", self.data_dir_name, "skill_memory")

        self.case_path = os.path.join(self.memory_dir, "case_memory.jsonl")
        self.case_partial_path = os.path.join(self.memory_dir, "case_memory.partial.jsonl")
        self.insight_path = os.path.join(self.memory_dir, "insight_memory.jsonl")
        self.risk_path = os.path.join(self.memory_dir, "risk_memory.jsonl")
        self.skill_stats_path = os.path.join(self.memory_dir, "skill_stats.json")
        self.manifest_path = os.path.join(self.memory_dir, "manifest.json")
        self.surface_embedding_path = os.path.join(
            self.memory_dir, "surface_clip_embeddings.json"
        )
        self.surface_ssr_path = (
            getattr(self.config, "surface_ssr_path", None)
            or os.path.join("data", self.data_dir_name, f"{self.data_dir_name}_SSR.jsonl")
        )

        self.case_memory: List[CaseMemoryItem] = []
        self.insight_memory: List[InsightMemoryItem] = []
        self.risk_memory: List[RiskMemoryItem] = []
        self.skill_stats: Dict[str, Dict[str, float]] = {}
        self.surface_embeddings: Dict[str, List[float]] = {}
        self.surface_embedding_metadata: Dict[str, Any] = {}
        self.surface_ssr: Dict[int, Tuple[List[int], List[float]]] = {}
        self._surface_ssr_loaded = False
        self._case_by_source_index: Dict[int, CaseMemoryItem] = {}
        self.is_loaded = False

    def load(self, build_if_missing: bool = True) -> bool:
        os.makedirs(self.memory_dir, exist_ok=True)
        has_all = (
            os.path.exists(self.case_path)
            and os.path.exists(self.insight_path)
            and os.path.exists(self.risk_path)
        )

        if build_if_missing and (not has_all or self._needs_rebuild()):
            self.build_from_unlabeled(use_llm=False)
            self.save()

        self.case_memory = [
            CaseMemoryItem.from_dict(row)
            for row in self._read_jsonl(self.case_path)
        ]
        self._case_by_source_index = {}
        self.insight_memory = [
            InsightMemoryItem.from_dict(row)
            for row in self._read_jsonl(self.insight_path)
        ]
        self.risk_memory = [
            RiskMemoryItem.from_dict(row)
            for row in self._read_jsonl(self.risk_path)
        ]
        if os.path.exists(self.skill_stats_path):
            with open(self.skill_stats_path, "r", encoding="utf-8") as f:
                self.skill_stats = json.load(f)
        self._load_surface_embedding_cache()
        self.is_loaded = True
        return True

    def save(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        self._write_jsonl(self.case_path, (item.to_dict() for item in self.case_memory))
        self._write_jsonl(
            self.insight_path, (item.to_dict() for item in self.insight_memory)
        )
        self._write_jsonl(self.risk_path, (item.to_dict() for item in self.risk_memory))
        with open(self.skill_stats_path, "w", encoding="utf-8") as f:
            json.dump(self.skill_stats, f, ensure_ascii=False, indent=2)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": MEMORY_SCHEMA_VERSION,
                    "dataset": self.dataset_name,
                    "label_free": True,
                    "notes": (
                        "Built without exposing gold labels or label-conditioned "
                        "reference explanations to planner/judge contexts."
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        self._save_surface_embedding_cache()

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _count_jsonl_rows(path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _load_partial_cases(self) -> Dict[str, CaseMemoryItem]:
        completed: Dict[str, CaseMemoryItem] = {}
        for row in self._read_jsonl(self.case_partial_path):
            try:
                item = CaseMemoryItem.from_dict(row)
            except TypeError:
                continue
            completed[item.source_id] = item
        if completed:
            print(
                f"Resuming MLLM memory build: loaded {len(completed)} "
                f"completed case annotations from {self.case_partial_path}"
            )
        return completed

    def _append_partial_case(self, case: CaseMemoryItem) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        with open(self.case_partial_path, "a", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False)
            f.write("\n")
            f.flush()

    def _needs_rebuild(self) -> bool:
        if not os.path.exists(self.manifest_path):
            return True
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return True
        return manifest.get("schema_version") != MEMORY_SCHEMA_VERSION

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "memory_dir": self.memory_dir,
            "is_loaded": self.is_loaded,
            "num_case_memory": len(self.case_memory),
            "num_insight_memory": len(self.insight_memory),
            "num_risk_memory": len(self.risk_memory),
            "num_skill_stats": len(self.skill_stats),
            "num_surface_embeddings": len(self.surface_embeddings),
            "surface_ssr_available": os.path.exists(self.surface_ssr_path),
            "num_surface_ssr_queries": (
                len(self.surface_ssr) if self._surface_ssr_loaded else None
            ),
        }

    def build_from_unlabeled(
        self,
        train_path: Optional[str] = None,
        use_llm: bool = False,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        max_cases: Optional[int] = None,
        use_existing_explanations: bool = False,
        resume: bool = True,
        checkpoint_every: int = 1,
    ) -> None:
        """Build memory from the unlabeled reference set.

        ``train_with_explanations.jsonl`` is used when available because it
        contains useful analysis text, but class labels are not copied into the
        memory context.  Set ``use_llm=True`` to let a multimodal model fill the
        structured fields from images; the deterministic fallback keeps local
        tests and ablations cheap.
        """
        if train_path is None:
            exp_path = os.path.join("data", self.data_dir_name, "train_with_explanations.jsonl")
            raw_path = os.path.join("data", self.data_dir_name, "train.jsonl")
            if use_existing_explanations and os.path.exists(exp_path):
                train_path = exp_path
            else:
                train_path = raw_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Reference set not found: {train_path}")

        caller = LLMCaller(client=client, model=model, component_name="MemoryBuilder")
        completed: Dict[str, CaseMemoryItem] = {}
        if use_llm and resume:
            completed = self._load_partial_cases()

        self.case_memory = []
        total_rows = self._count_jsonl_rows(train_path)
        if max_cases is not None:
            total_rows = min(total_rows, max_cases)
        iterator_desc = f"Building {self.dataset_name} case memory"
        with open(train_path, "r", encoding="utf-8") as f:
            iterator = tqdm(
                enumerate(f),
                total=total_rows,
                desc=iterator_desc,
                unit="case",
            )
            for idx, line in iterator:
                if max_cases is not None and idx >= max_cases:
                    break
                item = json.loads(line)
                case = self._case_from_item(
                    item,
                    idx,
                    keep_reference_explanation=use_existing_explanations,
                )
                if use_llm and case.source_id in completed:
                    self.case_memory.append(completed[case.source_id])
                    iterator.set_postfix({"resumed": len(completed)})
                    continue
                if use_llm:
                    image_path = self._resolve_image_path(case.image_filename)
                    annotated = self._annotate_case_with_llm(case, caller, image_path)
                    if annotated.metadata.get("llm_annotation_failed"):
                        # Keep a deterministic fallback case, but do not mark it
                        # completed in the partial file. A future resume can retry.
                        case = annotated
                    else:
                        case = annotated
                        self._append_partial_case(case)
                self.case_memory.append(case)

        self.insight_memory = self._build_insight_memory()
        self.risk_memory = self._build_risk_memory()

    def retrieve(
        self,
        query_signature: QuerySignature,
        caption: str = "",
        image_path: Optional[str] = None,
        query_index: Optional[int] = None,
        top_k_cases: Optional[int] = None,
        top_k_surface: Optional[int] = None,
        top_k_logic: Optional[int] = None,
        top_k_insights: int = 3,
        top_k_risks: int = 3,
    ) -> MemoryRetrievalResult:
        if not self.is_loaded:
            self.load(build_if_missing=True)

        top_k_cases = top_k_cases or getattr(self.config, "top_k_retrieval", 5)
        top_k_surface = top_k_surface or getattr(self.config, "surface_top_k", None) or top_k_cases
        top_k_logic = top_k_logic or getattr(self.config, "logic_top_k", None) or top_k_cases
        query_text = query_signature.to_retrieval_text(caption)
        surface_query_text = self._surface_query_text(query_signature, caption)

        logic_cases = self._rank_items(
            query_text,
            self.case_memory,
            top_k_logic,
            source="logic",
        )
        if getattr(self.config, "use_surface_retrieval", True):
            surface_cases, surface_diag = self._retrieve_surface_cases(
                query_text=surface_query_text,
                image_path=image_path,
                query_index=query_index,
                top_k=top_k_surface,
            )
        else:
            surface_cases, surface_diag = [], {
                "enabled": False,
                "reason": "disabled_by_config",
            }

        fused_cases = self._fuse_case_results(
            logic_cases=logic_cases,
            surface_cases=surface_cases,
            top_k=top_k_cases,
        )

        return MemoryRetrievalResult(
            query_text=query_text,
            case_memories=fused_cases,
            logic_case_memories=logic_cases,
            surface_case_memories=surface_cases,
            insight_memories=self._rank_items(
                query_text,
                self.insight_memory,
                top_k_insights,
                source="logic",
            ),
            risk_memories=self._rank_items(
                query_text,
                self.risk_memory,
                top_k_risks,
                source="logic",
            ),
            retrieval_diagnostics={
                "logic_top_k": top_k_logic,
                "surface_top_k": top_k_surface,
                "fused_top_k": top_k_cases,
                "surface": surface_diag,
                "fusion": {
                    "logic_weight": getattr(self.config, "logic_fusion_weight", 0.5),
                    "surface_weight": getattr(self.config, "surface_fusion_weight", 0.5),
                },
            },
        )

    def apply_update_decision(self, decision: MemoryUpdateDecision) -> None:
        changed = False
        if decision.write_case and decision.case_item:
            self.case_memory.append(decision.case_item)
            changed = True
        if decision.write_insight and decision.insight_item:
            self.insight_memory.append(decision.insight_item)
            changed = True
        if decision.write_risk and decision.risk_item:
            self.risk_memory.append(decision.risk_item)
            changed = True
        for skill, delta in decision.skill_reliability_updates.items():
            stats = self.skill_stats.setdefault(skill, {"reliability_delta": 0.0})
            stats["reliability_delta"] = stats.get("reliability_delta", 0.0) + delta
            changed = True
        if changed:
            self.save()

    def _case_from_item(
        self,
        item: Dict[str, Any],
        index: int,
        keep_reference_explanation: bool = False,
    ) -> CaseMemoryItem:
        image_key = self.dataset_config["image_key"]
        text_key = self.dataset_config["text_key"]

        source_id = str(item.get("id", item.get("train_index", index)))
        text = str(item.get(text_key, "") or "")
        explanation = item.get("explanation") if keep_reference_explanation else None
        memory_id = f"case_{self.dataset_name}_{source_id}"

        combined = f"{text} {explanation or ''}"
        mechanisms = self._detect_harm_mechanisms(combined)
        entities = self._extract_entities(combined)
        targets = self._extract_target_candidates(combined, mechanisms)

        supports_harmful, supports_harmless, uncertainty = self._heuristic_evidence(
            text=text,
            explanation=explanation or "",
            mechanisms=mechanisms,
            targets=targets,
        )

        return CaseMemoryItem(
            memory_id=memory_id,
            source_id=source_id,
            image_filename=item.get(image_key),
            text=text,
            visual_summary=item.get("image_description", ""),
            ocr_text=_as_list(item.get("ocr_text", [])),
            entities=entities,
            target_candidates=targets,
            possible_harm_mechanisms=mechanisms,
            supports_harmful=supports_harmful,
            supports_harmless=supports_harmless,
            uncertainty=uncertainty,
            prior_explanation=explanation,
            metadata={
                "dataset": self.dataset_name,
                "source_index": index,
                "label_hidden_for_memory": True,
                "reference_explanation_hidden": not keep_reference_explanation,
            },
        )

    def _resolve_image_path(self, image_filename: Optional[str]) -> Optional[str]:
        if not image_filename:
            return None
        image_path = os.path.join("data", self.data_dir_name, "images", image_filename)
        return image_path if os.path.exists(image_path) else None

    def _annotate_case_with_llm(
        self,
        case: CaseMemoryItem,
        caller: LLMCaller,
        image_path: Optional[str],
    ) -> CaseMemoryItem:
        prompt = f'''Analyze this meme as an unlabeled reference case.

Text: "{case.text}"

Return JSON only:
{{
  "visual_summary": "<concise image description>",
  "ocr_text": ["<text visible in image>"],
  "entities": ["<people/groups/objects/events>"],
  "target_candidates": ["<possible targets>"],
  "possible_harm_mechanisms": ["<stereotype/objectification/dehumanization/etc>"],
  "supports_harmful": ["<candidate evidence only, no final label>"],
  "supports_harmless": ["<candidate counter-evidence only, no final label>"],
  "uncertainty": ["<missing or ambiguous facts>"]
}}

Do not use or infer any gold label.'''
        response = caller.call(prompt, image_path=image_path, max_tokens=1200)
        if response.startswith("Error:"):
            case.metadata["llm_annotation_failed"] = True
            case.metadata["llm_error"] = response
            return case
        data = extract_json_object(response)
        if not data:
            case.metadata["llm_annotation_failed"] = True
            case.metadata["llm_error"] = "No JSON object found in MLLM response."
            return case
        case.visual_summary = data.get("visual_summary", case.visual_summary) or ""
        case.ocr_text = _as_list(data.get("ocr_text", case.ocr_text))
        case.entities = _as_list(data.get("entities", case.entities))
        case.target_candidates = _as_list(
            data.get("target_candidates", case.target_candidates)
        )
        case.possible_harm_mechanisms = _as_list(
            data.get("possible_harm_mechanisms", case.possible_harm_mechanisms)
        )
        case.supports_harmful = _as_list(
            data.get("supports_harmful", case.supports_harmful)
        )
        case.supports_harmless = _as_list(
            data.get("supports_harmless", case.supports_harmless)
        )
        case.uncertainty = _as_list(data.get("uncertainty", case.uncertainty))
        return case

    def _build_insight_memory(self) -> List[InsightMemoryItem]:
        groups: Dict[str, List[CaseMemoryItem]] = {}
        for case in self.case_memory:
            keys = case.possible_harm_mechanisms or ["general_boundary"]
            for key in keys:
                groups.setdefault(key, []).append(case)

        insights: List[InsightMemoryItem] = []
        for idx, (mechanism, cases) in enumerate(sorted(groups.items()), 1):
            if len(cases) < 2 and mechanism != "general_boundary":
                continue
            sample_texts = [c.text for c in cases[:5]]
            insights.append(
                InsightMemoryItem(
                    memory_id=f"insight_{self.dataset_name}_{idx:03d}",
                    title=_title_from_key(mechanism),
                    pattern=self._pattern_for_mechanism(mechanism, sample_texts),
                    applicable_when=self._applicability_for_mechanism(mechanism),
                    judgment_guidance=self._guidance_for_mechanism(mechanism),
                    related_case_ids=[c.memory_id for c in cases[:20]],
                    risk_notes=self._risk_notes_for_mechanism(mechanism),
                    reliability=0.55 + min(0.25, math.log(len(cases) + 1) / 20.0),
                )
            )

        if not insights:
            insights.append(
                InsightMemoryItem(
                    memory_id=f"insight_{self.dataset_name}_general",
                    title="General target and harm boundary",
                    pattern=(
                        "A meme should be judged by whether its image-text "
                        "combination promotes a harmful implication toward a "
                        "target, not by sarcasm or mismatch alone."
                    ),
                    applicable_when=["limited memory", "unclear harm mechanism"],
                    judgment_guidance=(
                        "First identify the target and harm mechanism. If both "
                        "are weak or absent, prefer the harmless claim."
                    ),
                    reliability=0.5,
                )
            )
        return insights

    def _build_risk_memory(self) -> List[RiskMemoryItem]:
        risks = [
            RiskMemoryItem(
                memory_id=f"risk_{self.dataset_name}_{idx:03d}",
                risk_type=row["risk_type"],
                description=row["description"],
                triggers=row["triggers"],
                recommended_skills=row["recommended_skills"],
                failure_mode=row["description"],
                reliability=0.65,
            )
            for idx, row in enumerate(STATIC_RISKS, 1)
        ]

        uncertain_cases = [case for case in self.case_memory if case.uncertainty]
        if uncertain_cases:
            risks.append(
                RiskMemoryItem(
                    memory_id=f"risk_{self.dataset_name}_uncertainty",
                    risk_type="case_level_uncertainty",
                    description=(
                        "Several reference cases contain unresolved target, "
                        "intent, or image-text relation ambiguity."
                    ),
                    triggers=sorted({
                        token
                        for case in uncertain_cases[:50]
                        for token in case.uncertainty[:2]
                    })[:10],
                    recommended_skills=[
                        "target_boundary",
                        "image_text_incongruity",
                        "counter_speech_quoted_criticism",
                    ],
                    failure_mode="Judge may overfit to one evidence side.",
                    reliability=0.55,
                )
            )
        return risks

    def _rank_items(
        self,
        query_text: str,
        items: Sequence[Any],
        top_k: int,
        source: str = "logic",
    ) -> List[ScoredMemory]:
        if not query_text or not items:
            return []
        docs = [item.retrieval_text() for item in items]
        scores = _tfidf_scores(query_text, docs)
        ranked = sorted(
            (
                ScoredMemory(item=item, score=score, source=source)
                for item, score in zip(items, scores)
            ),
            key=lambda row: row.score,
            reverse=True,
        )
        return [row for row in ranked[:top_k] if row.score > 0] or ranked[:top_k]

    def _retrieve_surface_cases(
        self,
        query_text: str,
        image_path: Optional[str],
        query_index: Optional[int],
        top_k: int,
    ) -> Tuple[List[ScoredMemory], Dict[str, Any]]:
        """Retrieve surface-similar cases.

        By default this uses precomputed SSR rows such as
        ``data/FHM/FHM_SSR.jsonl`` when available.  CLIP/text fallbacks remain
        available for datasets that do not provide SSR.
        """
        backend = str(
            getattr(self.config, "surface_retrieval_backend", "auto") or "auto"
        ).lower()
        diag: Dict[str, Any] = {
            "enabled": True,
            "backend": backend,
            "query_index": query_index,
        }

        if backend in {"none", "off", "disabled"}:
            diag["enabled"] = False
            diag["reason"] = "backend_disabled"
            return [], diag

        if backend in {"auto", "ssr", "precomputed", "precomputed_ssr"}:
            ssr_rows = self._retrieve_surface_cases_from_ssr(
                query_index=query_index,
                top_k=top_k,
                diagnostics=diag,
            )
            if ssr_rows is not None:
                return ssr_rows, diag
            if backend != "auto":
                return [], diag

        if backend in {"auto", "clip"}:
            clip_rows = self._retrieve_surface_cases_with_clip(
                query_text=query_text,
                image_path=image_path,
                top_k=top_k,
                diagnostics=diag,
            )
            if clip_rows is not None:
                return clip_rows, diag
            if backend == "clip":
                return [], diag

        docs = [item.surface_text() for item in self.case_memory]
        scores = _tfidf_scores(query_text, docs)
        ranked = sorted(
            (
                ScoredMemory(item=item, score=score, source="surface_text")
                for item, score in zip(self.case_memory, scores)
            ),
            key=lambda row: row.score,
            reverse=True,
        )
        diag["backend"] = "surface_text_tfidf"
        diag["fallback_reason"] = diag.get(
            "fallback_reason",
            "precomputed SSR/CLIP unavailable",
        )
        return ([row for row in ranked[:top_k] if row.score > 0] or ranked[:top_k]), diag

    def _retrieve_surface_cases_from_ssr(
        self,
        query_index: Optional[int],
        top_k: int,
        diagnostics: Dict[str, Any],
    ) -> Optional[List[ScoredMemory]]:
        if query_index is None:
            diagnostics["fallback_reason"] = "query_index missing for SSR"
            return None

        self._load_surface_ssr()
        if not self.surface_ssr:
            diagnostics["fallback_reason"] = f"SSR file unavailable or empty: {self.surface_ssr_path}"
            return None

        try:
            key = int(query_index)
        except (TypeError, ValueError):
            diagnostics["fallback_reason"] = f"invalid query_index for SSR: {query_index}"
            return None

        pair = self.surface_ssr.get(key)
        if not pair:
            diagnostics["fallback_reason"] = f"no SSR row for query index {key}"
            return None

        source_indices, scores = pair
        case_by_index = self._get_case_by_source_index()
        rows: List[ScoredMemory] = []
        missing = 0
        for source_index, score in zip(source_indices, scores):
            case = case_by_index.get(source_index)
            if not case:
                missing += 1
                continue
            rows.append(
                ScoredMemory(
                    item=case,
                    score=clamp_float(score, 0.0),
                    source="surface_ssr",
                )
            )
            if len(rows) >= top_k:
                break

        diagnostics["backend"] = "precomputed_ssr"
        diagnostics["ssr_path"] = self.surface_ssr_path
        diagnostics["ssr_candidates"] = len(source_indices)
        diagnostics["ssr_missing_cases"] = missing
        return rows

    def _load_surface_ssr(self) -> None:
        if self._surface_ssr_loaded:
            return
        self._surface_ssr_loaded = True
        self.surface_ssr = {}
        if not os.path.exists(self.surface_ssr_path):
            return

        for row in self._read_jsonl(self.surface_ssr_path):
            try:
                query_index = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            samples = row.get("samples") or []
            scores = row.get("scores") or []
            parsed_samples: List[int] = []
            parsed_scores: List[float] = []
            for position, sample in enumerate(samples):
                try:
                    parsed_samples.append(int(sample))
                except (TypeError, ValueError):
                    continue
                try:
                    score = float(scores[position])
                except (IndexError, TypeError, ValueError):
                    score = 1.0 / (position + 1)
                parsed_scores.append(score)
            if parsed_samples:
                self.surface_ssr[query_index] = (parsed_samples, parsed_scores)

    def _get_case_by_source_index(self) -> Dict[int, CaseMemoryItem]:
        if self._case_by_source_index:
            return self._case_by_source_index

        mapping: Dict[int, CaseMemoryItem] = {}
        for position, case in enumerate(self.case_memory):
            raw_index = case.metadata.get("source_index")
            if raw_index is None:
                raw_index = position
            try:
                source_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            mapping.setdefault(source_index, case)
        self._case_by_source_index = mapping
        return self._case_by_source_index

    def _retrieve_surface_cases_with_clip(
        self,
        query_text: str,
        image_path: Optional[str],
        top_k: int,
        diagnostics: Dict[str, Any],
    ) -> Optional[List[ScoredMemory]]:
        try:
            from PIL import Image
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            diagnostics["fallback_reason"] = f"missing dependency: {exc.name}"
            return None
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            diagnostics["fallback_reason"] = f"clip dependency unavailable: {exc}"
            return None

        if not image_path or not os.path.exists(image_path):
            diagnostics["fallback_reason"] = "query image missing"
            return None

        image_cases = [
            item for item in self.case_memory
            if item.image_filename and os.path.exists(self._resolve_image_path(item.image_filename) or "")
        ]
        if not image_cases:
            diagnostics["fallback_reason"] = "reference images missing"
            return None

        model_name = getattr(self.config, "clip_model_name", "clip-ViT-B-32")
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - model availability varies
            diagnostics["fallback_reason"] = f"could not load {model_name}: {exc}"
            return None

        self._ensure_surface_embeddings(model=model, image_cases=image_cases)
        if not self.surface_embeddings:
            diagnostics["fallback_reason"] = "surface embedding cache is empty"
            return None

        text_weight = float(getattr(self.config, "retrieval_weight_text", 0.2))
        image_weight = float(getattr(self.config, "retrieval_weight_image", 0.8))
        total_weight = max(1e-6, text_weight + image_weight)
        text_weight /= total_weight
        image_weight /= total_weight

        try:
            query_text_emb = _normalize_vector(model.encode(query_text).tolist())
            with Image.open(image_path) as image:
                query_image_emb = _normalize_vector(model.encode(image).tolist())
        except Exception as exc:  # pragma: no cover - image/model dependent
            diagnostics["fallback_reason"] = f"query embedding failed: {exc}"
            return None

        rows: List[ScoredMemory] = []
        for case in image_cases:
            cached = self.surface_embeddings.get(case.memory_id)
            if not cached:
                continue
            dim = len(cached) // 2
            if dim == 0:
                continue
            case_text_emb = cached[:dim]
            case_image_emb = cached[dim:]
            score = (
                text_weight * _dot(query_text_emb, case_text_emb)
                + image_weight * _dot(query_image_emb, case_image_emb)
            )
            rows.append(ScoredMemory(item=case, score=clamp_float(score, 0.0), source="surface_clip"))

        if not rows:
            diagnostics["fallback_reason"] = "no comparable CLIP rows"
            return None
        diagnostics["backend"] = "clip_image_text"
        diagnostics["clip_model"] = model_name
        diagnostics["num_reference_images"] = len(image_cases)
        return sorted(rows, key=lambda row: row.score, reverse=True)[:top_k]

    def _ensure_surface_embeddings(self, model: Any, image_cases: Sequence[CaseMemoryItem]) -> None:
        model_name = getattr(self.config, "clip_model_name", "clip-ViT-B-32")
        expected_hash = self._surface_cache_hash(model_name, image_cases)
        if (
            self.surface_embedding_metadata.get("cache_hash") == expected_hash
            and self.surface_embeddings
        ):
            return

        try:
            from PIL import Image
        except ModuleNotFoundError:
            return

        embeddings: Dict[str, List[float]] = {}
        for case in image_cases:
            image_path = self._resolve_image_path(case.image_filename)
            if not image_path:
                continue
            try:
                text_emb = _normalize_vector(model.encode(case.surface_text()).tolist())
                with Image.open(image_path) as image:
                    image_emb = _normalize_vector(model.encode(image).tolist())
                embeddings[case.memory_id] = text_emb + image_emb
            except Exception:
                continue

        self.surface_embeddings = embeddings
        self.surface_embedding_metadata = {
            "cache_hash": expected_hash,
            "clip_model": model_name,
            "num_embeddings": len(embeddings),
        }
        self._save_surface_embedding_cache()

    def _load_surface_embedding_cache(self) -> None:
        self.surface_embeddings = {}
        self.surface_embedding_metadata = {}
        if not os.path.exists(self.surface_embedding_path):
            return
        try:
            with open(self.surface_embedding_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.surface_embedding_metadata = data.get("metadata", {})
        self.surface_embeddings = {
            str(key): [float(v) for v in value]
            for key, value in (data.get("embeddings") or {}).items()
        }

    def _save_surface_embedding_cache(self) -> None:
        if not self.surface_embeddings and not self.surface_embedding_metadata:
            return
        with open(self.surface_embedding_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self.surface_embedding_metadata,
                    "embeddings": self.surface_embeddings,
                },
                f,
                ensure_ascii=False,
            )

    def _surface_cache_hash(
        self,
        model_name: str,
        image_cases: Sequence[CaseMemoryItem],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        digest.update(self.dataset_name.encode("utf-8"))
        for case in image_cases:
            digest.update(case.memory_id.encode("utf-8"))
            digest.update((case.image_filename or "").encode("utf-8"))
            digest.update(case.surface_text().encode("utf-8"))
        return digest.hexdigest()

    def _fuse_case_results(
        self,
        logic_cases: List[ScoredMemory],
        surface_cases: List[ScoredMemory],
        top_k: int,
    ) -> List[ScoredMemory]:
        logic_weight = float(getattr(self.config, "logic_fusion_weight", 0.5))
        surface_weight = float(getattr(self.config, "surface_fusion_weight", 0.5))
        fused: Dict[str, Dict[str, Any]] = {}

        for row in logic_cases:
            entry = fused.setdefault(row.item.memory_id, {"item": row.item, "logic": 0.0, "surface": 0.0})
            entry["logic"] = max(entry["logic"], row.score)
        for row in surface_cases:
            entry = fused.setdefault(row.item.memory_id, {"item": row.item, "logic": 0.0, "surface": 0.0})
            entry["surface"] = max(entry["surface"], row.score)

        rows = []
        for memory_id, entry in fused.items():
            score = logic_weight * entry["logic"] + surface_weight * entry["surface"]
            if entry["logic"] > 0 and entry["surface"] > 0:
                source = "fused_logic_surface"
            elif entry["surface"] > 0:
                source = "surface"
            else:
                source = "logic"
            rows.append(
                ScoredMemory(
                    item=entry["item"],
                    score=clamp_float(score, 0.0),
                    source=source,
                )
            )
        return sorted(rows, key=lambda row: row.score, reverse=True)[:top_k]

    @staticmethod
    def _surface_query_text(query_signature: QuerySignature, caption: str) -> str:
        parts = [
            caption,
            query_signature.visual_summary,
            " ".join(query_signature.ocr_text),
            " ".join(query_signature.entities),
        ]
        return " ".join(part for part in parts if part).strip() or query_signature.to_retrieval_text(caption)

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
        entities = []
        for token in tokens:
            clean = token.strip("'").lower()
            if clean in STOPWORDS or len(clean) < 4:
                continue
            if clean not in entities:
                entities.append(clean)
            if len(entities) >= 12:
                break
        return entities

    @staticmethod
    def _detect_harm_mechanisms(text: str) -> List[str]:
        lower = text.lower()
        mechanisms = []
        for mechanism, keywords in HARM_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", lower) for keyword in keywords):
                mechanisms.append(mechanism)
        if any(marker in lower for marker in ["not like", "but", "when", "me vs"]):
            mechanisms.append("image_text_incongruity")
        return mechanisms

    @staticmethod
    def _extract_target_candidates(text: str, mechanisms: List[str]) -> List[str]:
        lower = text.lower()
        targets = []
        for mechanism in mechanisms:
            for keyword in HARM_KEYWORDS.get(mechanism, []):
                if re.search(rf"\b{re.escape(keyword)}\b", lower):
                    targets.append(keyword)
        if any(word in lower for word in ["trump", "biden", "obama", "president"]):
            targets.append("public figure")
        return list(dict.fromkeys(targets))[:8]

    @staticmethod
    def _heuristic_evidence(
        text: str,
        explanation: str,
        mechanisms: List[str],
        targets: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        lower = f"{text} {explanation}".lower()
        supports_harmful = []
        supports_harmless = []
        uncertainty = []

        if targets and any(m in mechanisms for m in ["protected_group", "misogyny"]):
            supports_harmful.append(
                "The reference may involve a protected or gendered target."
            )
        if "violence" in mechanisms:
            supports_harmful.append("The reference contains violent framing.")
        if "historical_harm" in mechanisms or "health_or_disaster" in mechanisms:
            supports_harmful.append(
                "The reference may trivialize serious historical, health, or disaster harm."
            )
        if "sexualization" in mechanisms:
            supports_harmful.append("The reference contains sexualized framing.")

        harmless_markers = [
            "harmless",
            "positive",
            "encouragement",
            "does not target",
            "no harmful",
            "benign",
            "constructive",
        ]
        if any(marker in lower for marker in harmless_markers):
            supports_harmless.append(
                "The reference explanation emphasizes benign or non-targeting context."
            )
        if any(marker in lower for marker in ["critique", "criticize", "satire", "mocking hate"]):
            supports_harmless.append(
                "The reference may be critique or satire rather than promotion."
            )

        if "unclear" in lower or "ambiguous" in lower or not targets:
            uncertainty.append("Target or intent may be ambiguous.")
        if "image_text_incongruity" in mechanisms:
            uncertainty.append("Image-text relation needs verification.")

        return supports_harmful[:5], supports_harmless[:5], uncertainty[:5]

    @staticmethod
    def _pattern_for_mechanism(mechanism: str, sample_texts: List[str]) -> str:
        examples = "; ".join(sample_texts[:3])
        base = {
            "protected_group": (
                "Protected-group references require separating identity attack "
                "from neutral mention or anti-bigotry critique."
            ),
            "misogyny": (
                "Gendered humor becomes risky when it objectifies, demeans, or "
                "normalizes stereotypes about women."
            ),
            "violence": (
                "Violent wording is high risk when it targets a person or group, "
                "but quoted condemnation should be checked."
            ),
            "political": (
                "Political memes often sit near the public-figure criticism "
                "boundary; target and dangerous implication matter more than tone."
            ),
            "historical_harm": (
                "Historical harm references are risky when they trivialize "
                "atrocities or convert suffering into entertainment."
            ),
            "health_or_disaster": (
                "Health and disaster memes are risky when they normalize "
                "misinformation, schadenfreude, or crisis trivialization."
            ),
            "sexualization": (
                "Sexualized references are risky when they objectify a target or "
                "attach sexual framing to protected identities."
            ),
            "image_text_incongruity": (
                "Image-text mismatch is evidence only when it creates a harmful "
                "implication, not merely when it is funny or surprising."
            ),
        }.get(
            mechanism,
            "General cases require identifying target, intent, and harm mechanism.",
        )
        return f"{base} Representative texts: {examples}" if examples else base

    @staticmethod
    def _applicability_for_mechanism(mechanism: str) -> List[str]:
        return {
            "protected_group": ["identity terms", "group target", "stereotype"],
            "misogyny": ["women/girls mentioned", "gender stereotype", "objectification"],
            "violence": ["violent verbs", "threats", "celebration of harm"],
            "political": ["public figure", "government", "policy critique"],
            "historical_harm": ["war", "genocide", "slavery", "atrocity reference"],
            "health_or_disaster": ["disease", "pandemic", "disaster", "misfortune"],
            "sexualization": ["sexual innuendo", "objectification", "body framing"],
            "image_text_incongruity": ["caption-image mismatch", "irony", "contrast"],
        }.get(mechanism, ["unclear target", "unclear harm mechanism"])

    @staticmethod
    def _guidance_for_mechanism(mechanism: str) -> str:
        return {
            "protected_group": (
                "Check whether the group is the object of ridicule or whether the "
                "meme criticizes prejudice toward that group."
            ),
            "misogyny": (
                "Look for demeaning, objectifying, or stereotype-reinforcing "
                "framing; do not flag simple relationship humor without such cues."
            ),
            "violence": (
                "Treat calls for violence or celebration of harm as strong harmful "
                "evidence unless clearly quoted for condemnation."
            ),
            "political": (
                "Public-figure satire can be harmless; escalate only for "
                "dehumanization, protected-group spillover, dangerous claims, or "
                "celebration of suffering."
            ),
            "historical_harm": (
                "Distinguish educational or critical reference from trivialization "
                "or entertainment built from historical suffering."
            ),
            "health_or_disaster": (
                "Check whether humor spreads dangerous misinformation, dismisses "
                "severity, or celebrates misfortune."
            ),
            "sexualization": (
                "Ask whether sexual framing targets or objectifies someone rather "
                "than merely using adult innuendo."
            ),
            "image_text_incongruity": (
                "Use incongruity as a bridge to the implied message; it is not a "
                "standalone harmful signal."
            ),
        }.get(
            mechanism,
            "Identify the target and evidence relation before assigning weight.",
        )

    @staticmethod
    def _risk_notes_for_mechanism(mechanism: str) -> List[str]:
        return {
            "political": ["Political criticism can be over-flagged."],
            "image_text_incongruity": ["Surface mismatch can be over-weighted."],
            "protected_group": ["Anti-bigotry critique can quote harmful language."],
            "misogyny": ["Mild relationship jokes can be mistaken for misogyny."],
        }.get(mechanism, [])


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def _title_from_key(key: str) -> str:
    return key.replace("_", " ").title()


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "")
        if token.lower() not in STOPWORDS
    ]


def _tfidf_scores(query: str, docs: Sequence[str]) -> List[float]:
    """Dependency-light TF-IDF cosine similarity."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0 for _ in docs]

    doc_tokens = [_tokenize(doc) for doc in docs]
    df: Dict[str, int] = {}
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    num_docs = max(1, len(docs))

    def vector(tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        vec = {}
        for token, count in counts.items():
            idf = math.log((num_docs + 1) / (df.get(token, 0) + 1)) + 1.0
            vec[token] = count * idf
        return vec

    q_vec = vector(query_tokens)
    q_norm = math.sqrt(sum(value * value for value in q_vec.values())) or 1.0
    scores = []
    for tokens in doc_tokens:
        d_vec = vector(tokens)
        d_norm = math.sqrt(sum(value * value for value in d_vec.values())) or 1.0
        dot = sum(q_vec.get(token, 0.0) * d_vec.get(token, 0.0) for token in q_vec)
        scores.append(clamp_float(dot / (q_norm * d_norm), default=0.0))
    return scores
