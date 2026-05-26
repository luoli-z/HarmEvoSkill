# -*- coding: utf-8 -*-
"""
Query signature extraction for Module 2.

The signature is a compact, structured description of the current meme that
drives memory retrieval and skill planning.
"""
import os
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    OpenAI = None

from framework.config import DEFAULT_MODEL
from framework.llm_utils import LLMCallError, LLMCaller, extract_json_object
from framework.memory import HARM_KEYWORDS, QuerySignature


class QuerySignatureExtractor:
    """Extract a problem signature with an MLLM and a deterministic fallback."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        use_llm: bool = True,
        dataset_name: str = "FHM",
    ):
        self.use_llm = use_llm
        self.dataset_name = dataset_name
        self.dataset_signature_note = (
            "FHM calibration: do not output a final label, but when relevant expose "
            "implicit harmful mechanisms from the image-text combination: misogyny, "
            "women-as-object/domestic-appliance framing, sexual objectification, "
            "rape/consent trivialization, protected-group ridicule, racialized color/"
            "food/animal wording, religious or nationality stereotype, trans or "
            "disability mockery, human-animal comparison, victim mockery, violence "
            "normalization, terrorism/atrocity/tragedy humor, and coded euphemisms "
            "such as dishwasher/kitchen/sandwich, vegetable, ape/monkey/goat/pig, "
            "tampon/body jokes, or stop-resisting/police-violence jokes. Also note "
            "when the dominant reading is benign absurdity, a standalone animal/pet "
            "joke, ordinary family/relationship humor, historical description, "
            "counter-speech, or pure wordplay. For coded terms such as KFC, chocolate, "
            "vegetable, guide dog, gas, shower, diary, dishwasher, sandwich, or 'I'm down', "
            "briefly state whether the image-text pair anchors a target or leaves the code "
            "ambiguous; do not erase a plausible harmful mechanism when the meme implication is clear."
            if dataset_name.upper() == "FHM" else (
                "HarM calibration: do not output a final label, but when relevant "
                "include weak HarM mechanisms such as political cynicism, personal "
                "degradation of a named public figure, illness/diagnosis/death mockery, "
                "COVID-risk trivialization, public-health misinformation, hoax/fake-news/"
                "QAnon framing, vaccine/mask/testing/disinfectant/cure rhetoric, or "
                "disease-origin stigma. Also note harmless-looking mechanisms such as "
                "ordinary policy criticism, abstract virus jokes, quarantine coping, "
                "Corona/alcohol puns, and pure wordplay when those are dominant."
                if dataset_name.upper() == "HARM" else ""
            )
        )
        self.caller = LLMCaller(
            client=client,
            model=model,
            component_name="QuerySignature",
        )

    def extract(self, image_path: str, text: str) -> QuerySignature:
        if self.use_llm:
            signature = self._extract_with_llm(image_path, text)
            if signature:
                return signature
            raise LLMCallError("QuerySignature did not return valid JSON.")
        return self._fallback_signature(text)

    def _extract_with_llm(self, image_path: str, text: str) -> Optional[QuerySignature]:
        prompt = f'''Extract a structured problem signature for harmful meme analysis.

Caption / meme text: "{text}"

Return JSON only:
{{
  "visual_summary": "<concise image description>",
  "ocr_text": ["<text visible inside the image, excluding provided caption if duplicated>"],
  "entities": ["<people/groups/objects/events/concepts>"],
  "target_candidates": ["<who or what may be targeted>"],
  "tone": "<literal/ironic/sarcastic/satirical/aggressive/benign/unclear>",
  "image_text_relation": "<aligned/incongruent/contradictory/complementary/unclear>",
  "possible_harm_mechanisms": [
    "<protected_group_attack/misogyny/dehumanization/objectification/violence/trivialization/misinformation/counter_speech/etc>"
  ],
  "uncertainty": ["<missing or ambiguous facts>"]
}}

{self.dataset_signature_note}

Do not output a final harmful/harmless label.'''

        response = self.caller.call(
            prompt,
            image_path=image_path if os.path.exists(image_path) else None,
            max_tokens=1200,
        )
        data = extract_json_object(response)
        if not data:
            return None
        return QuerySignature(
            visual_summary=str(data.get("visual_summary", "") or ""),
            ocr_text=_as_list(data.get("ocr_text")),
            entities=_as_list(data.get("entities")),
            target_candidates=_as_list(data.get("target_candidates")),
            tone=str(data.get("tone", "unknown") or "unknown"),
            image_text_relation=str(
                data.get("image_text_relation", "unknown") or "unknown"
            ),
            possible_harm_mechanisms=_as_list(
                data.get("possible_harm_mechanisms")
            ),
            uncertainty=_as_list(data.get("uncertainty")),
            raw=data,
        )

    @staticmethod
    def _fallback_signature(text: str) -> QuerySignature:
        lower = (text or "").lower()
        entities: List[str] = []
        mechanisms: List[str] = []
        targets: List[str] = []

        for mechanism, keywords in HARM_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", lower):
                    if mechanism not in mechanisms:
                        mechanisms.append(mechanism)
                    if keyword not in targets:
                        targets.append(keyword)
                    if keyword not in entities:
                        entities.append(keyword)

        if any(marker in lower for marker in ["not", "but", "when", "me vs", "expectation"]):
            mechanisms.append("image_text_incongruity")

        tone = "unclear"
        if any(word in lower for word in ["beat", "kill", "hate", "stupid"]):
            tone = "aggressive"
        elif any(word in lower for word in ["lol", "when", "me when"]):
            tone = "humorous"

        uncertainty = []
        if not targets:
            uncertainty.append("target is not explicit from text alone")
        uncertainty.append("visual content was not parsed by fallback extractor")

        return QuerySignature(
            visual_summary="",
            ocr_text=[],
            entities=entities[:12],
            target_candidates=targets[:8],
            tone=tone,
            image_text_relation="unknown",
            possible_harm_mechanisms=list(dict.fromkeys(mechanisms)),
            uncertainty=uncertainty,
            raw={"fallback": True},
        )


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
