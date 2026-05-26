# -*- coding: utf-8 -*-
"""
Graph-assisted judgment and label-free memory update for Module 3.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    OpenAI = None

from framework.config import DATASET_CONFIGS, DEFAULT_FRAMEWORK_CONFIG, DEFAULT_MODEL
from framework.boundary_rules import get_harm_boundary_rules
from framework.evidence_graph import ClaimEvidenceGraph
from framework.llm_utils import LLMCallError, LLMCaller, clamp_float, extract_json_object, normalize_prediction
from framework.memory import (
    CaseMemoryItem,
    InsightMemoryItem,
    MemoryRetrievalResult,
    MemoryUpdateDecision,
    QuerySignature,
    RiskMemoryItem,
)
from framework.planner import SkillPlan
from framework.skills import EvidencePacket


@dataclass
class GraphJudgmentResult:
    prediction: int
    confidence: float
    reasoning_summary: str
    key_evidence: List[str]
    graph_diagnosis: Dict[str, Any]
    claim_scores: Dict[str, float]
    memory_update_decision: Optional[MemoryUpdateDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "key_evidence": self.key_evidence,
            "graph_diagnosis": self.graph_diagnosis,
            "claim_scores": self.claim_scores,
            "memory_update_decision": (
                self.memory_update_decision.to_dict()
                if self.memory_update_decision else None
            ),
        }


class GraphJudge:
    """Judge that reasons over a harmful/harmless claim-evidence graph."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        config: Optional[Any] = None,
        dataset_name: str = "FHM",
        use_llm: bool = True,
    ):
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.dataset_name = dataset_name
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        self.use_llm = use_llm
        self.caller = LLMCaller(client=client, model=model, component_name="GraphJudge")
        self.boundary_rules = get_harm_boundary_rules(dataset_name)
        self.dataset_calibration_note = (
            "FHM reminder: do not over-correct toward harmless. FHM harmful memes "
            "are often terse, implicit image-text jokes. If the caption and image "
            "together ground misogyny, protected-group ridicule, racial/religious/"
            "nationality stereotyping, trans/disability mockery, human-animal "
            "comparison, sexual objectification, rape/consent trivialization, "
            "violence normalization, terrorism/atrocity humor, or victim mockery, "
            "support Claim_Harmful even without an explicit slur or threat. Still "
            "keep ordinary slapstick, pets/standalone animals, family/ex-partner "
            "jokes, generic arrest jokes, historical descriptions, identity/religion/"
            "nationality puns, anti-hate or perpetrator-directed jokes, and pure "
            "absurdity harmless when no such mechanism is grounded. Before choosing "
            "Claim_Harmful for a weak or coded case, name the protected/vulnerable "
            "target and the exact degrading proposition; if either is missing, prefer "
            "Claim_Harmless. For borderline FHM cases, compare mechanisms rather "
            "than keywords: pun vs contempt, object/food wordplay vs human "
            "dehumanization, historical reference vs victim mockery, generic sex "
            "joke vs gendered sexual degradation, and standalone animal/pet joke "
            "vs human-animal comparison. Do not demand perfect explicitness when "
            "the offensive implication is clear from the image-text pair."
            if dataset_name.upper() == "FHM" else (
                "HarM reminder: treat 'somewhat harmful' as a weak positive class. "
                "A meme can be Claim_Harmful when it degrades a named person, "
                "organization, community, or society through political/COVID humor, "
                "even without hate speech or a direct dangerous instruction. Do not "
                "dismiss Claim_Harmful only because the content is satire, public-"
                "figure criticism, or counter-speech. Also do not overfire on COVID "
                "keywords alone: ordinary policy criticism, abstract virus jokes, "
                "Corona/alcohol puns, quarantine coping, and pure wordplay remain "
                "Claim_Harmless unless there is personal degradation, illness/death "
                "mockery, stigma, persuasive misinformation, or risky-behavior "
                "normalization."
                if dataset_name.upper() == "HARM" else ""
            )
        )
        self.dataset_output_note = (
            'For HarM, also include "harm_severity": "none", "somewhat_harmful", '
            'or "very_harmful". If the meme fits somewhat_harmful, prediction must '
            f'be "{self.dataset_config.get("positive_label", "harmful")}".'
            if dataset_name.upper() == "HARM" else ""
        )
        self.dataset_decision_note = (
            "For FHM, harmful stance or implication can be implicit but must be "
            "grounded in the image-text pair. Ask whether the meme's joke depends "
            "on degrading a protected group/person, women as a class, a disabled "
            "or trans target, a victim, or a person through animalization, sexual "
            "objectification, violence, or serious suffering. If yes, do not dismiss "
            "it as mere absurdity, dark humor, or ambiguity. If the image-text pair "
            "only creates benign absurdity, ordinary relationship/family humor, a "
            "standalone animal/pet joke, or non-targeted slapstick, choose Claim_Harmless."
            " Also choose Claim_Harmless for generic identity puns, religious/nationality "
            "wordplay, historical/documentary captions, anti-hate commentary, or jokes "
            "whose target is a perpetrator/hate group/terrorist/ideology rather than a "
            "protected victim, unless the meme clearly generalizes inferiority, criminality, "
            "sexual degradation, animalization, threat, or victim mockery to a protected target."
            " For close FHM cases, decide the mechanism pair briefly: pun vs contempt, "
            "object/food wordplay vs human dehumanization, historical reference vs victim "
            "mockery, generic sex joke vs gendered sexual degradation, and standalone animal/"
            "pet joke vs human-animal comparison. Avoid over-ruling strong harmful cues with "
            "excessive uncertainty."
            if dataset_name.upper() == "FHM" else (
                "For HarM, harmful stance or implication includes \"somewhat harmful\" "
                "public-health or political harm. It can be weak. Ask: who is the "
                "joke on, and what is the main punchline doing? Claim_Harmful can "
                "be supported when the meme makes a named person/organization/"
                "community look stupid, reckless, diseased, unfit, corrupt, or "
                "deserving ridicule through COVID/political framing; when illness, "
                "diagnosis, death, case counts, masks, testing, or pandemic response "
                "becomes personal humiliation; or when hoax/fake-news/QAnon, vaccine, "
                "mask, disinfectant/bleach, or cure rhetoric is made socially persuasive. "
                "Do not require explicit endorsement, hate speech, or a direct instruction. "
                "If the evidence only shows sensitive-topic mention, historical description, "
                "documentary/technical framing, political criticism, animal harm, or "
                "denunciation of hate/violence, choose Claim_Harmless and explain the "
                "boundary. For HarM, do this only after ruling out personal degradation, "
                "concrete public-health harm, misinformation amplification, stigma, and "
                "illness/death/vulnerability ridicule. Valid HarM harmless exceptions "
                "include ordinary policy accountability, abstract virus personification, "
                "generic pandemic inconvenience/coping, neutral public-health reminders, "
                "Corona/alcohol puns, one-off absurdity that does not persuade or degrade, "
                "and clear debunking/condemnation without personal humiliation."
                if dataset_name.upper() == "HARM" else ""
            )
        )

    def judge(
        self,
        image_path: str,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
        skill_plan: SkillPlan,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
    ) -> GraphJudgmentResult:
        if self.use_llm:
            result = self._judge_with_llm(
                image_path,
                text,
                query_signature,
                memory_result,
                skill_plan,
                evidence_packets,
                evidence_graph,
            )
            if result:
                return result
            raise LLMCallError("GraphJudge did not return valid JSON.")
        return self._fallback_judge(evidence_packets, evidence_graph)

    def _judge_with_llm(
        self,
        image_path: str,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
        skill_plan: SkillPlan,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
    ) -> Optional[GraphJudgmentResult]:
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")
        packets_json = json.dumps(
            [packet.to_dict() for packet in evidence_packets],
            ensure_ascii=False,
            indent=2,
        )
        graph_json = json.dumps(evidence_graph.to_dict(), ensure_ascii=False, indent=2)
        prompt = f'''You are the final graph-assisted judge for harmful meme analysis.

Target meme text: "{text}"
[Image attached if available]

Query signature:
{json.dumps(query_signature.to_dict(), ensure_ascii=False, indent=2)}

Retrieved memory:
{memory_result.get_context_string(top_k_cases=5, top_k_insights=3, top_k_risks=3)}

Skill plan:
{json.dumps(skill_plan.to_dict(), ensure_ascii=False, indent=2)}

Evidence packets:
{packets_json}

Claim-evidence graph:
{graph_json}

Decision boundary guidance:
{self.boundary_rules}

Judge by comparing Claim_Harmful and Claim_Harmless. First discard noisy
evidence, then identify the strongest evidence and conflicts. Use graph
relations, not free-form intuition alone.
Before predicting Claim_Harmful, explicitly verify that the target meme has:
1. a target,
2. a harmful stance or implication toward that target,
3. concrete image/text evidence for endorsement, ridicule, dehumanization,
   objectification, stereotyping, threat, or normalization.

{self.dataset_decision_note}

When a retrieved memory case looks harmful, do not copy its conclusion unless
the target meme has the same decisive harmful mechanism. Memory is analogy and
contrast evidence, not a label vote.

{self.dataset_calibration_note}

{self.dataset_output_note}

Dataset labels:
- Claim_Harmful corresponds to "{positive_label}"
- Claim_Harmless corresponds to "{negative_label}"

Return JSON only:
{{
  "claim_scores": {{"Claim_Harmful": <0.0-1.0>, "Claim_Harmless": <0.0-1.0>}},
  "boundary_check": {{
    "joke_target": "<who or what the joke is mainly on>",
    "harm_mechanism": "<protected-group attack / misogyny / objectification / dehumanization / violence / victim mockery / misinformation / none>",
    "harmless_exception": "<benign absurdity / standalone animal or pet joke / ordinary family or relationship joke / historical description / counter-speech / pure wordplay / none>"
  }},
  "graph_diagnosis": {{
    "decisive_evidence_nodes": ["<evidence ids or skill names>"],
    "noisy_or_weak_evidence": ["<evidence ids or descriptions>"],
    "conflicts": ["<conflict descriptions>"],
    "missing_information": ["<uncertainties>"],
    "skill_reliability_notes": {{"<skill_key>": "<note>"}}
  }},
  "prediction": "{positive_label}" or "{negative_label}",
  "harm_severity": "none" or "somewhat_harmful" or "very_harmful",
  "confidence": <0.0-1.0>,
  "reasoning_summary": "<3-5 sentence summary based on the graph>",
  "key_evidence": ["<most important evidence>"]
}}'''
        response = self.caller.call(
            prompt,
            image_path=image_path if os.path.exists(image_path) else None,
            max_tokens=1800,
        )
        data = extract_json_object(response)
        if not data:
            return None
        prediction = normalize_prediction(
            data.get("prediction", ""),
            positive_label=positive_label,
        )
        scores = data.get("claim_scores") or {}
        return GraphJudgmentResult(
            prediction=prediction,
            confidence=clamp_float(data.get("confidence", 0.5), 0.5),
            reasoning_summary=str(data.get("reasoning_summary", response)),
            key_evidence=_as_list(data.get("key_evidence")),
            graph_diagnosis=data.get("graph_diagnosis", {}),
            claim_scores={
                "Claim_Harmful": clamp_float(scores.get("Claim_Harmful", 0.5), 0.5),
                "Claim_Harmless": clamp_float(scores.get("Claim_Harmless", 0.5), 0.5),
            },
        )

    def _fallback_judge(
        self,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
    ) -> GraphJudgmentResult:
        summary = evidence_graph.summarize()
        harmful_score = (
            summary["support_harmful"]
            + summary["weaken_harmless"] * 0.5
            - summary["weaken_harmful"] * 0.6
        )
        harmless_score = (
            summary["support_harmless"]
            + summary["weaken_harmful"] * 0.5
            - summary["weaken_harmless"] * 0.6
        )
        prediction = 1 if harmful_score >= harmless_score else 0
        margin = abs(harmful_score - harmless_score)
        total = abs(harmful_score) + abs(harmless_score) + 1e-6
        confidence = clamp_float(0.5 + 0.45 * (margin / total), 0.55)

        sorted_packets = sorted(evidence_packets, key=lambda p: p.confidence, reverse=True)
        key_evidence = []
        for packet in sorted_packets:
            key_evidence.extend(packet.evidence[:1])
            if len(key_evidence) >= 3:
                break

        conflicts = [
            edge.to_dict()
            for edge in evidence_graph.edges
            if edge.relation == "conflicts_with"
        ][:5]
        missing = [
            uncertainty
            for packet in evidence_packets
            for uncertainty in packet.uncertainty
        ][:5]
        return GraphJudgmentResult(
            prediction=prediction,
            confidence=confidence,
            reasoning_summary=(
                "Fallback graph judge compared weighted support and weakening "
                "relations for Claim_Harmful and Claim_Harmless. "
                f"Scores were harmful={harmful_score:.2f}, "
                f"harmless={harmless_score:.2f}."
            ),
            key_evidence=key_evidence,
            graph_diagnosis={
                "decisive_evidence_nodes": [
                    packet.skill for packet in sorted_packets[:3]
                ],
                "noisy_or_weak_evidence": [],
                "conflicts": conflicts,
                "missing_information": missing,
                "skill_reliability_notes": {},
            },
            claim_scores={
                "Claim_Harmful": harmful_score,
                "Claim_Harmless": harmless_score,
            },
        )


class MemoryCurator:
    """Creates label-free memory update proposals from graph diagnosis."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        dataset_name: str = "FHM",
        use_llm: bool = False,
    ):
        self.dataset_name = dataset_name
        self.use_llm = use_llm
        self.caller = LLMCaller(client=client, model=model, component_name="MemoryCurator")

    def propose_update(
        self,
        text: str,
        image_path: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
        judgment: GraphJudgmentResult,
        sample_index: Optional[int] = None,
    ) -> MemoryUpdateDecision:
        if self.use_llm:
            decision = self._propose_with_llm(
                text,
                image_path,
                query_signature,
                memory_result,
                evidence_packets,
                evidence_graph,
                judgment,
                sample_index,
            )
            if decision:
                return decision
        return self._fallback_update(
            text,
            image_path,
            query_signature,
            evidence_packets,
            evidence_graph,
            judgment,
            sample_index,
        )

    def _propose_with_llm(
        self,
        text: str,
        image_path: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
        judgment: GraphJudgmentResult,
        sample_index: Optional[int],
    ) -> Optional[MemoryUpdateDecision]:
        prompt = f'''Create a label-free memory update proposal from this judgment.

Text: "{text}"
Query signature:
{json.dumps(query_signature.to_dict(), ensure_ascii=False, indent=2)}

Retrieved memory ids: {memory_result.get_memory_ids()}

Evidence graph summary:
{json.dumps(evidence_graph.summarize(), ensure_ascii=False, indent=2)}

Judgment:
{json.dumps(judgment.to_dict(), ensure_ascii=False, indent=2)}

Rules:
- High confidence, evidence-supported, non-conflicting cases may become pseudo case memory.
- Do not require zero uncertainty for case memory; memes often have ambiguous intent/context.
  Prefer case memory when decisive evidence is present and uncertainty is not the main reason for the judgment.
- Reusable patterns may become insight memory.
- Low confidence, conflicts, or missing key evidence should become risk memory.
- Avoid writing risk memory merely because several uncertainty strings exist;
  prefer risk memory for substantial ambiguity or explicit evidence conflict.
- Do not include gold labels.

Return JSON only:
{{
  "write_case": true/false,
  "case_summary": {{
    "possible_harm_mechanisms": ["..."],
    "supports_harmful": ["..."],
    "supports_harmless": ["..."],
    "uncertainty": ["..."]
  }},
  "write_insight": true/false,
  "insight": {{
    "title": "...",
    "pattern": "...",
    "applicable_when": ["..."],
    "judgment_guidance": "...",
    "risk_notes": ["..."],
    "reliability": <0.0-1.0>
  }},
  "write_risk": true/false,
  "risk": {{
    "risk_type": "...",
    "description": "...",
    "triggers": ["..."],
    "recommended_skills": ["..."],
    "failure_mode": "...",
    "reliability": <0.0-1.0>
  }},
  "skill_reliability_updates": {{"<skill_key>": <delta between -0.05 and 0.05>}},
  "rationale": "..."
}}'''
        response = self.caller.call(
            prompt,
            image_path=image_path if os.path.exists(image_path) else None,
            max_tokens=1600,
        )
        data = extract_json_object(response)
        if not data:
            return None
        return self._decision_from_data(data, text, query_signature, sample_index)

    def _fallback_update(
        self,
        text: str,
        image_path: str,
        query_signature: QuerySignature,
        evidence_packets: List[EvidencePacket],
        evidence_graph: ClaimEvidenceGraph,
        judgment: GraphJudgmentResult,
        sample_index: Optional[int],
    ) -> MemoryUpdateDecision:
        summary = evidence_graph.summarize()
        conflicts = summary["num_conflicts"]
        uncertainties = [
            item for packet in evidence_packets for item in packet.uncertainty
        ]

        source_id = str(sample_index if sample_index is not None else "online")
        memory_id = f"case_{self.dataset_name}_pseudo_{source_id}"
        supports_harmful = [
            ev for packet in evidence_packets
            if "Claim_Harmful" in packet.supports
            for ev in packet.evidence
        ][:5]
        supports_harmless = [
            ev for packet in evidence_packets
            if "Claim_Harmless" in packet.supports
            for ev in packet.evidence
        ][:5]
        evidence_count = len(supports_harmful) + len(supports_harmless)
        uncertainty_count = len(set(uncertainties))
        support_harmful = float(summary.get("support_harmful", 0.0)) + float(
            summary.get("weaken_harmless", 0.0)
        )
        support_harmless = float(summary.get("support_harmless", 0.0)) + float(
            summary.get("weaken_harmful", 0.0)
        )
        dominant_support = max(support_harmful, support_harmless)
        support_margin = abs(support_harmful - support_harmless)
        used_fallback_judge = judgment.reasoning_summary.startswith("Fallback graph judge")
        uncertainty_is_tolerable = (
            uncertainty_count <= 3
            or (judgment.confidence >= 0.85 and evidence_count >= 3)
        )
        if self.dataset_name.upper() == "FHM":
            case_confidence_threshold = 0.9
            case_margin_threshold = 0.85
            case_uncertainty_limit = 1
        else:
            case_confidence_threshold = 0.75
            case_margin_threshold = 0.25
            case_uncertainty_limit = 999

        write_case = (
            judgment.confidence >= case_confidence_threshold
            and not used_fallback_judge
            and conflicts == 0
            and evidence_count >= 2
            and uncertainty_is_tolerable
            and uncertainty_count <= case_uncertainty_limit
            and dominant_support >= 0.75
            and support_margin >= case_margin_threshold
        )

        case_item = CaseMemoryItem(
            memory_id=memory_id,
            source_id=source_id,
            image_filename=os.path.basename(image_path) if image_path else None,
            text=text,
            visual_summary=query_signature.visual_summary,
            ocr_text=query_signature.ocr_text,
            entities=query_signature.entities,
            target_candidates=query_signature.target_candidates,
            possible_harm_mechanisms=query_signature.possible_harm_mechanisms,
            supports_harmful=supports_harmful,
            supports_harmless=supports_harmless,
            uncertainty=uncertainties[:5],
            prior_explanation=judgment.reasoning_summary,
            metadata={
                "pseudo": True,
                "confidence": judgment.confidence,
                "prediction_hidden_for_memory": True,
                "case_write_policy": "gated_high_confidence_evidence_supported",
                "evidence_count": evidence_count,
                "uncertainty_count": uncertainty_count,
                "support_harmful": support_harmful,
                "support_harmless": support_harmless,
                "support_margin": support_margin,
            },
        )

        risk_item = None
        balanced_conflict = (
            support_harmful >= 0.75
            and support_harmless >= 0.75
            and abs(support_harmful - support_harmless) <= 0.5
        )
        missing_decisive_evidence = (
            uncertainty_count >= 4
            and evidence_count < 2
        )
        diagnosis = judgment.graph_diagnosis or {}
        diagnosis_missing = diagnosis.get("missing_information") or []
        judge_reports_missing_decisive = (
            len(diagnosis_missing) >= 3
            and judgment.confidence < 0.75
        )
        write_risk = (
            judgment.confidence < 0.6
            or conflicts > 0
            or balanced_conflict
            or missing_decisive_evidence
            or judge_reports_missing_decisive
        )
        if write_risk:
            risk_reasons = []
            if judgment.confidence < 0.6:
                risk_reasons.append("low_confidence")
            if conflicts > 0:
                risk_reasons.append("graph_conflict")
            if balanced_conflict:
                risk_reasons.append("balanced_harmful_harmless_evidence")
            if missing_decisive_evidence:
                risk_reasons.append("missing_decisive_evidence")
            if judge_reports_missing_decisive:
                risk_reasons.append("judge_reports_missing_information")
            risk_item = RiskMemoryItem(
                memory_id=f"risk_{self.dataset_name}_online_{source_id}",
                risk_type="online_gated_risk",
                description=(
                    "Online judge found a gated risk condition: low confidence, "
                    "explicit graph conflict, balanced harmful/harmless evidence, "
                    "or missing decisive evidence."
                ),
                triggers=risk_reasons + (uncertainties[:5] or query_signature.possible_harm_mechanisms[:5]),
                recommended_skills=[
                    packet.skill for packet in evidence_packets
                    if packet.uncertainty or not packet.evidence
                ][:4] or [packet.skill for packet in evidence_packets[:3]],
                failure_mode=judgment.reasoning_summary,
                reliability=0.55,
            )

        skill_updates: Dict[str, float] = {}
        for packet in evidence_packets:
            if packet.confidence >= 0.75 and packet.evidence:
                skill_updates[packet.skill] = skill_updates.get(packet.skill, 0.0) + 0.01
            if not packet.evidence or packet.uncertainty:
                skill_updates[packet.skill] = skill_updates.get(packet.skill, 0.0) - 0.005

        return MemoryUpdateDecision(
            write_case=write_case,
            write_insight=False,
            write_risk=write_risk,
            case_item=case_item if write_case else None,
            risk_item=risk_item,
            skill_reliability_updates=skill_updates,
            rationale=(
                "Fallback curator writes high-confidence, evidence-supported, "
                "non-conflicting judgments as pseudo case memory and writes risk "
                "memory only when gated low-confidence, conflict, balanced-evidence, "
                "or missing-evidence conditions are met."
            ),
        )

    def _decision_from_data(
        self,
        data: Dict[str, Any],
        text: str,
        query_signature: QuerySignature,
        sample_index: Optional[int],
    ) -> MemoryUpdateDecision:
        source_id = str(sample_index if sample_index is not None else "online")
        case_summary = data.get("case_summary") or {}
        case_item = CaseMemoryItem(
            memory_id=f"case_{self.dataset_name}_pseudo_{source_id}",
            source_id=source_id,
            image_filename=None,
            text=text,
            visual_summary=query_signature.visual_summary,
            ocr_text=query_signature.ocr_text,
            entities=query_signature.entities,
            target_candidates=query_signature.target_candidates,
            possible_harm_mechanisms=_as_list(
                case_summary.get(
                    "possible_harm_mechanisms",
                    query_signature.possible_harm_mechanisms,
                )
            ),
            supports_harmful=_as_list(case_summary.get("supports_harmful")),
            supports_harmless=_as_list(case_summary.get("supports_harmless")),
            uncertainty=_as_list(case_summary.get("uncertainty")),
            metadata={"pseudo": True, "prediction_hidden_for_memory": True},
        )

        insight_data = data.get("insight") or {}
        insight_item = None
        if insight_data:
            insight_item = InsightMemoryItem(
                memory_id=f"insight_{self.dataset_name}_online_{source_id}",
                title=str(insight_data.get("title", "Online induced insight")),
                pattern=str(insight_data.get("pattern", "")),
                applicable_when=_as_list(insight_data.get("applicable_when")),
                judgment_guidance=str(insight_data.get("judgment_guidance", "")),
                risk_notes=_as_list(insight_data.get("risk_notes")),
                reliability=clamp_float(insight_data.get("reliability", 0.55), 0.55),
            )

        risk_data = data.get("risk") or {}
        risk_item = None
        if risk_data:
            risk_item = RiskMemoryItem(
                memory_id=f"risk_{self.dataset_name}_online_{source_id}",
                risk_type=str(risk_data.get("risk_type", "online_risk")),
                description=str(risk_data.get("description", "")),
                triggers=_as_list(risk_data.get("triggers")),
                recommended_skills=_as_list(risk_data.get("recommended_skills")),
                failure_mode=str(risk_data.get("failure_mode", "")),
                reliability=clamp_float(risk_data.get("reliability", 0.55), 0.55),
            )

        return MemoryUpdateDecision(
            write_case=bool(data.get("write_case", False)),
            write_insight=bool(data.get("write_insight", False)) and insight_item is not None,
            write_risk=bool(data.get("write_risk", False)) and risk_item is not None,
            case_item=case_item,
            insight_item=insight_item,
            risk_item=risk_item,
            skill_reliability_updates={
                str(key): float(value)
                for key, value in (data.get("skill_reliability_updates") or {}).items()
            },
            rationale=str(data.get("rationale", "")),
        )


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
