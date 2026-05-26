# -*- coding: utf-8 -*-
"""
Main pipeline for the skill-memory harmful meme framework.

The pipeline follows the three modules in ``方法模块.md``:

Module 1: Label-free Skill-Memory Construction
Module 2: Memory-guided Skill Planning and Evidence Generation
Module 3: Graph-assisted Judgment and Self-evolving Memory Update
"""
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    OpenAI = None
try:
    import openai as legacy_openai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    legacy_openai = None
from tqdm import tqdm

from framework.config import (
    API_BASE_URL,
    DATASET_CONFIGS,
    DEFAULT_FRAMEWORK_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_PATH_CONFIG,
    FrameworkConfig,
)
from framework.evidence_graph import ClaimEvidenceGraph, EvidenceGraphBuilder
from framework.judge import GraphJudge, GraphJudgmentResult, MemoryCurator
from framework.llm_utils import LLMCallError
from framework.memory import MemoryRetrievalResult, QuerySignature, ScoredMemory, SkillMemoryStore
from framework.planner import SkillPlan, SkillPlanner
from framework.signature import QuerySignatureExtractor
from framework.skills import (
    EvidencePacket,
    DEFAULT_SKILL_CARDS,
    ReasoningSkillExecutor,
    SkillCard,
    SkillInducer,
    SkillRepository,
    SkillValidator,
    SkillType,
)


@dataclass
class PipelineResult:
    """Complete result from the skill-memory pipeline."""

    sample_index: int
    image_path: str
    text: str
    actual_label: Optional[int]

    query_signature: QuerySignature
    memory_result: MemoryRetrievalResult
    skill_plan: SkillPlan
    evidence_packets: List[EvidencePacket]
    evidence_graph: ClaimEvidenceGraph
    judgment_result: GraphJudgmentResult

    predicted_label: int
    confidence: float
    reasoning: str
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.sample_index,
            "image_path": self.image_path,
            "text": self.text,
            "actual": self.actual_label,
            "predicted": self.predicted_label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "query_signature": self.query_signature.to_dict(),
            "retrieved_memory": self.memory_result.to_dict(),
            "selected_skills": [skill.value for skill in self.skill_plan.selected_skills],
            "skill_plan": self.skill_plan.to_dict(),
            "evidence_packets": [packet.to_dict() for packet in self.evidence_packets],
            "claim_evidence_graph": self.evidence_graph.to_dict(),
            "graph_diagnosis": self.judgment_result.graph_diagnosis,
            "claim_scores": self.judgment_result.claim_scores,
            "key_evidence": self.judgment_result.key_evidence,
            "memory_update_decision": (
                self.judgment_result.memory_update_decision.to_dict()
                if self.judgment_result.memory_update_decision else None
            ),
            "processing_time": self.processing_time,
        }

    def is_correct(self) -> Optional[bool]:
        if self.actual_label is None:
            return None
        return self.predicted_label == self.actual_label


class MemeDetectionPipeline:
    """
    Skill-memory pipeline for zero-shot / label-free harmful meme detection.
    """

    def __init__(
        self,
        dataset_name: str,
        config: Optional[FrameworkConfig] = None,
        model: str = DEFAULT_MODEL,
        use_memory: bool = True,
        preload_memory: bool = True,
        use_llm_signature: bool = True,
        use_llm_planner: bool = True,
        use_llm_skills: bool = True,
        use_llm_judge: bool = True,
        use_llm_curator: bool = False,
        enable_memory_update: bool = True,
        memory_dir: Optional[str] = None,
        skill_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.model = model
        self.use_memory = use_memory
        self.enable_memory_update = enable_memory_update
        self.memory_dir = memory_dir
        self.skill_dir = skill_dir or memory_dir

        if OpenAI is not None:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=API_BASE_URL,
            )
        else:
            self.client = None
            if (
                legacy_openai is None
                and not os.getenv("OPENAI_API_KEY")
                and any([use_llm_signature, use_llm_planner, use_llm_skills, use_llm_judge, use_llm_curator])
            ):
                raise RuntimeError(
                    "LLM-enabled pipeline runs require OPENAI_API_KEY. "
                    "Set it or pass deterministic/fallback options."
                )

        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        self.path_config = DEFAULT_PATH_CONFIG
        self.base_path = self.path_config.get_dataset_path(dataset_name)
        self.image_base_path = self.path_config.get_image_path(dataset_name)

        self.memory_store: Optional[SkillMemoryStore] = None
        if use_memory:
            self.memory_store = SkillMemoryStore(
                dataset_name,
                config=self.config,
                memory_dir=self.memory_dir,
            )
            if preload_memory:
                self.memory_store.load(build_if_missing=True)
                print(f"Skill memory loaded: {self.memory_store.get_statistics()}")

        self.skill_repository = SkillRepository(dataset_name, skill_dir=self.skill_dir)
        self.skill_repository.load(build_if_missing=True)

        self.signature_extractor = QuerySignatureExtractor(
            client=self.client,
            model=model,
            use_llm=use_llm_signature,
            dataset_name=dataset_name,
        )
        self.skill_planner = SkillPlanner(
            client=self.client,
            model=model,
            config=self.config,
            use_llm=use_llm_planner,
            dataset_name=dataset_name,
        )
        self.skill_executor = ReasoningSkillExecutor(
            client=self.client,
            model=model,
            use_llm=use_llm_skills,
            dataset_name=dataset_name,
        )
        self.graph_builder = EvidenceGraphBuilder()
        self.judge = GraphJudge(
            client=self.client,
            model=model,
            config=self.config,
            dataset_name=dataset_name,
            use_llm=use_llm_judge,
        )
        self.memory_curator = MemoryCurator(
            client=self.client,
            model=model,
            dataset_name=dataset_name,
            use_llm=use_llm_curator,
        )

    def build_memory(
        self,
        train_path: Optional[str] = None,
        use_llm: bool = False,
        max_cases: Optional[int] = None,
        induce_skills: bool = True,
        resume: bool = True,
    ) -> Dict[str, Any]:
        if not self.memory_store:
            self.memory_store = SkillMemoryStore(
                self.dataset_name,
                config=self.config,
                memory_dir=self.memory_dir,
            )
        self.memory_store.build_from_unlabeled(
            train_path=train_path,
            use_llm=use_llm,
            client=self.client,
            model=self.model,
            max_cases=max_cases,
            resume=resume,
        )
        self.memory_store.save()
        self.memory_store.is_loaded = True

        if induce_skills:
            memory_context = MemoryRetrievalResult(
                query_text="offline skill induction",
                case_memories=[],
                insight_memories=[
                    ScoredMemory(item=item, score=1.0, source="offline_induction")
                    for item in self.memory_store.insight_memory[:8]
                ],
                risk_memories=[
                    ScoredMemory(item=item, score=1.0, source="offline_induction")
                    for item in self.memory_store.risk_memory[:8]
                ],
            ).get_context_string(top_k_insights=8, top_k_risks=8)
            inducer = SkillInducer(
                client=self.client,
                model=self.model,
                enabled=use_llm,
            )
            seed_cards = list(DEFAULT_SKILL_CARDS.values())
            induced_cards = [
                card for card in inducer.induce(seed_cards, memory_context)
                if SkillValidator.validate_card(card)
            ]
            self.skill_repository.skills = {
                card.skill_type: card for card in induced_cards
            }
            self.skill_repository.save()

        return self.memory_store.get_statistics()

    def process_single(
        self,
        image_path: str,
        text: str,
        sample_index: int = 0,
        actual_label: Optional[int] = None,
        use_memory: bool = True,
    ) -> PipelineResult:
        start_time = time.time()

        query_signature = self.signature_extractor.extract(image_path, text)

        if use_memory and self.use_memory and self.memory_store:
            if not self.memory_store.is_loaded:
                self.memory_store.load(build_if_missing=True)
            memory_result = self.memory_store.retrieve(
                query_signature=query_signature,
                caption=text,
                image_path=image_path,
                query_index=sample_index,
                top_k_cases=self.config.top_k_retrieval,
            )
        else:
            memory_result = MemoryRetrievalResult(query_text=query_signature.to_retrieval_text(text))

        skill_cards = self.skill_repository.all()
        skill_plan = self.skill_planner.plan(
            image_path=image_path,
            text=text,
            query_signature=query_signature,
            memory_result=memory_result,
            skill_cards=skill_cards,
        )
        selected_cards = [
            self.skill_repository.get(skill)
            for skill in skill_plan.execution_order
        ]

        evidence_packets = self.skill_executor.execute_many(
            selected_cards,
            image_path=image_path,
            text=text,
            query_signature=query_signature,
            memory_result=memory_result,
            parallel=self.config.parallel_tool_execution,
            max_workers=self.config.max_workers,
        )

        evidence_graph = self.graph_builder.build(
            evidence_packets,
            memory_result=memory_result,
        )
        judgment_result = self.judge.judge(
            image_path=image_path,
            text=text,
            query_signature=query_signature,
            memory_result=memory_result,
            skill_plan=skill_plan,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
        )

        update_decision = self.memory_curator.propose_update(
            text=text,
            image_path=image_path,
            query_signature=query_signature,
            memory_result=memory_result,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
            judgment=judgment_result,
            sample_index=sample_index,
        )
        judgment_result.memory_update_decision = update_decision

        if self.enable_memory_update and self.memory_store and update_decision:
            applied_decision = self._filter_memory_update_decision(update_decision)
            self.memory_store.apply_update_decision(applied_decision)
            self.skill_repository.apply_reliability_updates(
                applied_decision.skill_reliability_updates
            )

        processing_time = time.time() - start_time
        return PipelineResult(
            sample_index=sample_index,
            image_path=image_path,
            text=text,
            actual_label=actual_label,
            query_signature=query_signature,
            memory_result=memory_result,
            skill_plan=skill_plan,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
            judgment_result=judgment_result,
            predicted_label=judgment_result.prediction,
            confidence=judgment_result.confidence,
            reasoning=judgment_result.reasoning_summary,
            processing_time=processing_time,
        )

    def _filter_memory_update_decision(self, decision: Any) -> Any:
        policy = str(getattr(self.config, "memory_update_policy", "active") or "active").lower()
        filtered = copy.deepcopy(decision)
        if policy == "active":
            return decision
        if policy == "proposal_only":
            filtered.write_case = False
            filtered.write_insight = False
            filtered.write_risk = False
            filtered.skill_reliability_updates = {}
            return filtered
        if policy == "risk_only":
            filtered.write_case = False
            filtered.write_insight = False
            filtered.skill_reliability_updates = {}
            return filtered
        if policy == "no_case":
            filtered.write_case = False
            return filtered
        return decision

    def process_dataset(
        self,
        test_jsonl_path: Optional[str] = None,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        start_from: int = 0,
        use_memory: bool = True,
        show_progress: bool = True,
        resume_output: bool = True,
        overwrite_output: bool = False,
    ) -> Dict[str, Any]:
        if test_jsonl_path is None:
            test_jsonl_path = os.path.join(self.base_path, "test.jsonl")

        with open(test_jsonl_path, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        if max_samples:
            test_data = test_data[start_from:start_from + max_samples]
        else:
            test_data = test_data[start_from:]
        indexed_data = [
            (start_from + idx_offset, item)
            for idx_offset, item in enumerate(test_data)
        ]

        if output_path is None:
            results_dir = self.path_config.get_results_path(self.dataset_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"skill_memory_{timestamp}.jsonl")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._prepare_output_path(
            output_path=output_path,
            resume_output=resume_output,
            overwrite_output=overwrite_output,
        )

        results: List[PipelineResult] = []
        correct_count = 0
        total_count = 0
        all_actual: List[int] = []
        all_predicted: List[int] = []
        processing_times: List[float] = []
        completed_indices = set()
        existing_has_summary = False
        error_indices: set = set()

        if resume_output:
            state = self._load_existing_output_state(
                output_path=output_path,
                valid_indices={idx for idx, _ in indexed_data},
            )
            completed_indices = state["completed_indices"]
            correct_count = state["correct_count"]
            total_count = state["total_count"]
            all_actual = state["all_actual"]
            all_predicted = state["all_predicted"]
            processing_times = state["processing_times"]
            existing_has_summary = state["has_summary"]
            error_indices = state.get("error_indices", set())
            if completed_indices:
                print(
                    f"Resuming output: found {len(completed_indices)} completed "
                    f"samples in {output_path}"
                )

        pending_data = [
            (sample_index, item)
            for sample_index, item in indexed_data
            if sample_index not in completed_indices
        ]
        self._ensure_output_trailing_newline(output_path)

        iterator = (
            tqdm(
                pending_data,
                total=len(pending_data),
                desc=(
                    f"Processing (resume {len(completed_indices)} done)"
                    if completed_indices else "Processing"
                ),
            )
            if show_progress else pending_data
        )

        with open(output_path, "a", encoding="utf-8") as f_out:
            for sample_index, item in iterator:
                image_filename, text, label = self._get_item_data(item)
                if not image_filename or text is None:
                    continue

                image_path = os.path.join(self.image_base_path, image_filename)
                if not os.path.exists(image_path):
                    print(
                        f"Warning: Image not found, using text-only fallback: {image_path}"
                    )

                try:
                    result = self.process_single(
                        image_path=image_path,
                        text=text,
                        sample_index=sample_index,
                        actual_label=label,
                        use_memory=use_memory,
                    )
                except LLMCallError as exc:
                    print(f"API error processing sample {sample_index}: {exc}")
                    error_indices.add(sample_index)
                    error_dict = {
                        "index": sample_index,
                        "image_path": image_path,
                        "text": text,
                        "actual": label,
                        "error": {
                            "type": "LLMCallError",
                            "message": str(exc),
                        },
                        "status": "error",
                    }
                    json.dump(error_dict, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    f_out.flush()
                    continue
                except Exception as exc:
                    print(f"Error processing sample {sample_index}: {exc}")
                    continue

                results.append(result)
                error_indices.discard(sample_index)
                total_count += 1
                all_actual.append(label)
                all_predicted.append(result.predicted_label)
                processing_times.append(result.processing_time)
                if result.is_correct():
                    correct_count += 1

                result_dict = result.to_dict()
                result_dict["ratio"] = [total_count, correct_count]
                json.dump(result_dict, f_out, ensure_ascii=False)
                f_out.write("\n")
                f_out.flush()

                if show_progress:
                    acc = correct_count / total_count if total_count else 0.0
                    iterator.set_postfix({"acc": f"{acc:.4f}"})

        summary = self._summarize_results(
            all_actual=all_actual,
            all_predicted=all_predicted,
            total_count=total_count,
            correct_count=correct_count,
            results=results,
            processing_times=processing_times,
            output_path=output_path,
            use_memory=use_memory,
            unresolved_error_count=len(error_indices),
            unresolved_error_indices=sorted(error_indices),
        )

        if pending_data or not existing_has_summary:
            with open(output_path, "a", encoding="utf-8") as f_out:
                json.dump({"summary": summary}, f_out, ensure_ascii=False)
                f_out.write("\n")

        print(f"\n=== Results for {self.dataset_name} ===")
        print(f"Accuracy: {summary['accuracy']:.4f} ({correct_count}/{total_count})")
        print(f"Macro F1: {summary['macro_f1']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"Results saved to: {output_path}")
        return summary

    @staticmethod
    def _ensure_output_trailing_newline(output_path: str) -> None:
        if not output_path or not os.path.exists(output_path):
            return
        if os.path.getsize(output_path) == 0:
            return
        with open(output_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last_byte = f.read(1)
        if last_byte not in {b"\n", b"\r"}:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write("\n")

    @staticmethod
    def _prepare_output_path(
        output_path: str,
        resume_output: bool,
        overwrite_output: bool,
    ) -> None:
        if not output_path or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return
        if overwrite_output:
            with open(output_path, "w", encoding="utf-8"):
                pass
            print(f"Overwriting existing output: {output_path}")
            return
        if not resume_output:
            raise FileExistsError(
                f"Output file already exists and is non-empty: {output_path}. "
                "Use a new --output_path, remove --no_resume_output to resume, "
                "or add --overwrite_output to intentionally rerun from scratch."
            )

    def _load_existing_output_state(
        self,
        output_path: str,
        valid_indices: Optional[set] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "completed_indices": set(),
            "correct_count": 0,
            "total_count": 0,
            "all_actual": [],
            "all_predicted": [],
            "processing_times": [],
            "has_summary": False,
            "error_indices": set(),
        }
        if not output_path or not os.path.exists(output_path):
            return state

        rows_by_index: Dict[int, Dict[str, Any]] = {}
        with open(output_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: skipping malformed result line {line_no} in {output_path}"
                    )
                    continue
                if "summary" in row:
                    state["has_summary"] = True
                    continue
                if row.get("status") == "error" or row.get("error"):
                    try:
                        sample_index = int(row.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if valid_indices is None or sample_index in valid_indices:
                        state["error_indices"].add(sample_index)
                    continue
                try:
                    sample_index = int(row.get("index"))
                except (TypeError, ValueError):
                    continue
                if valid_indices is not None and sample_index not in valid_indices:
                    continue
                rows_by_index[sample_index] = row

        for sample_index in sorted(rows_by_index):
            row = rows_by_index[sample_index]
            state["completed_indices"].add(sample_index)
            state["error_indices"].discard(sample_index)
            actual = row.get("actual")
            predicted = row.get("predicted")
            if actual is not None and predicted is not None:
                try:
                    actual_i = int(actual)
                    predicted_i = int(predicted)
                except (TypeError, ValueError):
                    actual_i = predicted_i = None
                if actual_i is not None and predicted_i is not None:
                    state["all_actual"].append(actual_i)
                    state["all_predicted"].append(predicted_i)
                    state["total_count"] += 1
                    if actual_i == predicted_i:
                        state["correct_count"] += 1
            try:
                state["processing_times"].append(float(row.get("processing_time")))
            except (TypeError, ValueError):
                pass
        return state

    def _get_item_data(self, item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
        image_key = self.dataset_config["image_key"]
        text_key = self.dataset_config["text_key"]
        label_key = self.dataset_config["label_key"]
        label_mapping = self.dataset_config.get("label_mapping")

        image_filename = item.get(image_key)
        text = item.get(text_key)
        raw_label = item.get(label_key)

        if isinstance(raw_label, list):
            raw_label = raw_label[0] if raw_label else None
        if label_mapping:
            mapped = label_mapping.get(raw_label)
            if mapped is None:
                mapped = label_mapping.get(str(raw_label).strip().lower())
            if mapped is None:
                mapped = 0 if str(raw_label).strip().lower() in {"0", "false", "no", "not harmful", "harmless", "non-offensive", "non-offensiv"} else 1
            label = int(mapped)
        else:
            label = int(raw_label) if raw_label is not None else 0
        return image_filename, text, label

    def _summarize_results(
        self,
        all_actual: List[int],
        all_predicted: List[int],
        total_count: int,
        correct_count: int,
        results: List[PipelineResult],
        processing_times: Optional[List[float]],
        output_path: str,
        use_memory: bool,
        unresolved_error_count: int = 0,
        unresolved_error_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        accuracy = correct_count / total_count if total_count else 0.0
        if all_actual:
            from sklearn.metrics import f1_score, precision_score, recall_score

            macro_f1 = f1_score(all_actual, all_predicted, average="macro", zero_division=0)
            precision = precision_score(
                all_actual, all_predicted, average="macro", zero_division=0
            )
            recall = recall_score(
                all_actual, all_predicted, average="macro", zero_division=0
            )
        else:
            macro_f1 = precision = recall = 0.0

        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
        else:
            avg_time = (
                sum(result.processing_time for result in results) / len(results)
                if results else 0.0
            )
        return {
            "dataset": self.dataset_name,
            "model": self.model,
            "total_samples": total_count,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "average_processing_time": avg_time,
            "unresolved_error_count": unresolved_error_count,
            "unresolved_error_indices": unresolved_error_indices or [],
            "use_memory": use_memory,
            "top_k_memory": self.config.top_k_retrieval,
            "use_surface_retrieval": self.config.use_surface_retrieval,
            "surface_retrieval_backend": self.config.surface_retrieval_backend,
            "enable_memory_update": self.enable_memory_update,
            "memory_update_policy": getattr(self.config, "memory_update_policy", "active"),
            "memory_dir": self.memory_store.memory_dir if self.memory_store else None,
            "skill_dir": self.skill_repository.skill_dir,
            "output_path": output_path,
        }


class AblationPipeline(MemeDetectionPipeline):
    """Ablation variants for the new skill-memory method."""

    def __init__(
        self,
        dataset_name: str,
        config: Optional[FrameworkConfig] = None,
        model: str = DEFAULT_MODEL,
        use_memory: bool = True,
        use_planning: bool = True,
        use_graph: bool = True,
        use_all_skills: bool = False,
        specific_skills: Optional[List[SkillType]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_memory=use_memory,
            preload_memory=use_memory,
            **kwargs,
        )
        self.use_planning = use_planning
        self.use_graph = use_graph
        self.use_all_skills = use_all_skills
        self.specific_skills = specific_skills

    def process_single(
        self,
        image_path: str,
        text: str,
        sample_index: int = 0,
        actual_label: Optional[int] = None,
        use_memory: bool = True,
    ) -> PipelineResult:
        if self.use_planning and not self.use_all_skills and not self.specific_skills:
            return super().process_single(
                image_path=image_path,
                text=text,
                sample_index=sample_index,
                actual_label=actual_label,
                use_memory=use_memory,
            )

        start_time = time.time()
        query_signature = self.signature_extractor.extract(image_path, text)

        if use_memory and self.use_memory and self.memory_store:
            if not self.memory_store.is_loaded:
                self.memory_store.load(build_if_missing=True)
            memory_result = self.memory_store.retrieve(
                query_signature=query_signature,
                caption=text,
                image_path=image_path,
                query_index=sample_index,
                top_k_cases=self.config.top_k_retrieval,
            )
        else:
            memory_result = MemoryRetrievalResult(query_text=query_signature.to_retrieval_text(text))

        if self.use_all_skills:
            selected = [card.skill_type for card in self.skill_repository.all()]
        elif self.specific_skills:
            selected = self.specific_skills
        else:
            selected = [
                SkillType.TARGET_BOUNDARY,
                SkillType.IMAGE_TEXT_INCONGRUITY,
                SkillType.PROTECTED_GROUP_ATTRIBUTION,
            ]

        skill_plan = SkillPlan(
            selected_skills=selected,
            execution_order=selected,
            reasoning="Ablation fixed skill set.",
            memory_patterns=[],
            confidence=1.0,
        )
        selected_cards = [self.skill_repository.get(skill) for skill in selected]
        evidence_packets = self.skill_executor.execute_many(
            selected_cards,
            image_path=image_path,
            text=text,
            query_signature=query_signature,
            memory_result=memory_result,
            parallel=self.config.parallel_tool_execution,
            max_workers=self.config.max_workers,
        )
        evidence_graph = self.graph_builder.build(evidence_packets, memory_result)
        judgment_result = self.judge.judge(
            image_path=image_path,
            text=text,
            query_signature=query_signature,
            memory_result=memory_result,
            skill_plan=skill_plan,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
        )
        update_decision = self.memory_curator.propose_update(
            text=text,
            image_path=image_path,
            query_signature=query_signature,
            memory_result=memory_result,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
            judgment=judgment_result,
            sample_index=sample_index,
        )
        judgment_result.memory_update_decision = update_decision

        return PipelineResult(
            sample_index=sample_index,
            image_path=image_path,
            text=text,
            actual_label=actual_label,
            query_signature=query_signature,
            memory_result=memory_result,
            skill_plan=skill_plan,
            evidence_packets=evidence_packets,
            evidence_graph=evidence_graph,
            judgment_result=judgment_result,
            predicted_label=judgment_result.prediction,
            confidence=judgment_result.confidence,
            reasoning=judgment_result.reasoning_summary,
            processing_time=time.time() - start_time,
        )


def run_ablation_study(dataset_name: str, max_samples: int = 100) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    configurations = [
        ("full", {"use_memory": True, "use_planning": True}),
        ("no_memory", {"use_memory": False, "use_planning": True}),
        ("no_planning_default_skills", {"use_memory": True, "use_planning": False}),
        ("all_skills", {"use_memory": True, "use_planning": False, "use_all_skills": True}),
    ]
    for config_name, params in configurations:
        print(f"\n=== Running ablation: {config_name} ===")
        pipeline = AblationPipeline(dataset_name=dataset_name, **params)
        results[config_name] = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=f"results/{dataset_name}/skill_ablation_{config_name}.jsonl",
        )
    return results


if __name__ == "__main__":
    pipeline = MemeDetectionPipeline(
        dataset_name="FHM",
        model="gemini-flash",
        use_memory=True,
        use_llm_signature=False,
        use_llm_planner=False,
        use_llm_skills=False,
        use_llm_judge=False,
    )
    stats = pipeline.memory_store.get_statistics() if pipeline.memory_store else {}
    print(f"Pipeline initialized. Memory stats: {stats}")
