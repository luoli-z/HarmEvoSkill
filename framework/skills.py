# -*- coding: utf-8 -*-
"""
Reasoning skills for harmful meme analysis.

This file replaces the old fixed "tools" abstraction with reusable skill cards.
Each skill is a reasoning capability that consumes the query meme plus retrieved
case/insight/risk memory and emits an evidence packet, not a final label.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    OpenAI = None

from framework.config import API_BASE_URL, DEFAULT_MODEL, canonical_data_name
from framework.boundary_rules import get_harm_boundary_rules
from framework.llm_utils import LLMCallError, LLMCaller, clamp_float, extract_json_object
from framework.memory import MemoryRetrievalResult, QuerySignature


class SkillType(Enum):
    TARGET_BOUNDARY = "target_boundary"
    POLITICAL_CRITICISM_BOUNDARY = "political_criticism_boundary"
    PROTECTED_GROUP_ATTRIBUTION = "protected_group_attribution"
    MISOGYNISTIC_FRAMING = "misogynistic_framing"
    HISTORICAL_HARM_TRIVIALIZATION = "historical_harm_trivialization"
    IMAGE_TEXT_INCONGRUITY = "image_text_incongruity"
    CULTURAL_TEMPLATE_INTERPRETATION = "cultural_template_interpretation"
    COUNTER_SPEECH_QUOTED_CRITICISM = "counter_speech_quoted_criticism"


@dataclass
class SkillCard:
    """A callable and refinable reasoning skill."""

    skill_type: SkillType
    name: str
    purpose: str
    when_to_use: List[str]
    reasoning_steps: List[str]
    positive_indicators: List[str]
    negative_indicators: List[str]
    verifier: List[str]
    reliability: float = 0.65
    source_memory_ids: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.skill_type.value

    def to_prompt_block(self) -> str:
        return f'''### {self.name} ({self.key})
Purpose: {self.purpose}
When to use: {"; ".join(self.when_to_use)}
Reasoning steps: {"; ".join(self.reasoning_steps)}
Indicators for harmful claim: {"; ".join(self.positive_indicators)}
Indicators for harmless claim: {"; ".join(self.negative_indicators)}
Verifier: {"; ".join(self.verifier)}
Reliability: {self.reliability:.2f}'''

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_type": self.skill_type.value,
            "name": self.name,
            "purpose": self.purpose,
            "when_to_use": self.when_to_use,
            "reasoning_steps": self.reasoning_steps,
            "positive_indicators": self.positive_indicators,
            "negative_indicators": self.negative_indicators,
            "verifier": self.verifier,
            "reliability": self.reliability,
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCard":
        payload = dict(data)
        payload["skill_type"] = SkillType(payload["skill_type"])
        return cls(**payload)


@dataclass
class EvidencePacket:
    """Structured output from one skill."""

    skill: str
    local_claim: str
    supports: List[str]
    weakens: List[str]
    evidence: List[str]
    uncertainty: List[str] = field(default_factory=list)
    confidence: float = 0.5
    memory_support: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def suggests_harmful(self) -> Optional[bool]:
        if "Claim_Harmful" in self.supports and "Claim_Harmless" not in self.supports:
            return True
        if "Claim_Harmless" in self.supports and "Claim_Harmful" not in self.supports:
            return False
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "local_claim": self.local_claim,
            "supports": self.supports,
            "weakens": self.weakens,
            "evidence": self.evidence,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "memory_support": self.memory_support,
            "raw": self.raw,
        }


DEFAULT_SKILL_CARDS: Dict[SkillType, SkillCard] = {
    SkillType.TARGET_BOUNDARY: SkillCard(
        skill_type=SkillType.TARGET_BOUNDARY,
        name="Target Boundary Skill",
        purpose=(
            "Identify who or what is targeted and whether the target boundary "
            "supports harmful or harmless interpretation."
        ),
        when_to_use=[
            "target is ambiguous",
            "meme may attack a person, group, institution, or self",
            "retrieved risk memory mentions target boundary confusion",
        ],
        reasoning_steps=[
            "List explicit and implicit target candidates",
            "Classify each as protected group, public figure, institution, concept, individual, or self",
            "Decide whether the meme attacks the target or merely comments on it",
        ],
        positive_indicators=[
            "mockery or attack directed at a protected group",
            "dehumanization or celebration of suffering",
            "the joke depends on humiliating a vulnerable target",
        ],
        negative_indicators=[
            "target is a public institution or policy",
            "self-deprecation",
            "no clear target or only benign commentary",
        ],
        verifier=[
            "Do not call something harmful only because a target exists",
            "Check whether target class changes the strength of evidence",
        ],
    ),
    SkillType.POLITICAL_CRITICISM_BOUNDARY: SkillCard(
        skill_type=SkillType.POLITICAL_CRITICISM_BOUNDARY,
        name="Political Criticism Boundary Skill",
        purpose=(
            "Distinguish legitimate political criticism or public-figure satire "
            "from harmful attack."
        ),
        when_to_use=[
            "politicians, governments, elections, or policies appear",
            "risk memory mentions political satire boundary",
        ],
        reasoning_steps=[
            "Identify whether the target is public power or a protected identity",
            "Check whether the meme criticizes conduct/policy or personally degrades a named actor",
            "For HarM, separate ordinary accountability from weak individual harm such as humiliation, incompetence framing, illness mockery, or political cynicism",
            "Look for misinformation, dehumanization, or celebration/trivialization of illness/death",
        ],
        positive_indicators=[
            "encourages violence or dehumanization",
            "spreads dangerous misinformation",
            "celebrates illness, death, or disaster",
            "frames a named politician, public figure, or organization as stupid, reckless, corrupt, unfit, diseased, or personally ridiculous",
            "reduces serious democratic or public-health choices to cynical/flippant meme criteria",
            "uses COVID illness, diagnosis, case counts, masks, vaccines, testing, or pandemic response failure as a political punchline",
            "turns hoax, fake-news, disinfectant, or risky public-health rhetoric into memorable satire",
        ],
        negative_indicators=[
            "criticizes policy, leadership, preparedness, spending, delay, or public conduct without personal degradation",
            "satirizes a public figure without protected-group spillover or public-health harm",
            "does not promote dangerous claims",
            "COVID is incidental rather than the main joke frame",
            "joke target is an abstract virus, generic pandemic frustration, or pure political wordplay",
        ],
        verifier=[
            "Public figure target is not automatically harmless",
            "Political tone is not automatically harmful",
        ],
    ),
    SkillType.PROTECTED_GROUP_ATTRIBUTION: SkillCard(
        skill_type=SkillType.PROTECTED_GROUP_ATTRIBUTION,
        name="Protected Group Attribution Skill",
        purpose=(
            "Determine whether the meme attributes negative traits or attacks to "
            "a protected group."
        ),
        when_to_use=[
            "race, gender, religion, nationality, sexuality, disability, or ethnicity is mentioned",
            "retrieved memory flags group attribution risk",
        ],
        reasoning_steps=[
            "Identify protected categories mentioned visually or textually",
            "Separate mention from attribution",
            "Determine whether a negative property, stereotype, threat, or insult is assigned to the group",
        ],
        positive_indicators=[
            "negative stereotype is generalized to a group",
            "group is portrayed as inferior, dangerous, dirty, criminal, or deserving harm",
            "slurs or coded language target the group",
        ],
        negative_indicators=[
            "group is mentioned neutrally",
            "meme criticizes prejudice against the group",
            "target is an individual for non-identity conduct",
        ],
        verifier=[
            "Evidence must connect the protected identity to the harmful implication",
            "Do not infer protected-group attack from identity words alone",
        ],
    ),
    SkillType.MISOGYNISTIC_FRAMING: SkillCard(
        skill_type=SkillType.MISOGYNISTIC_FRAMING,
        name="Misogynistic Framing Skill",
        purpose=(
            "Identify woman-targeted demeaning, objectifying, stereotype-based, "
            "or anti-feminist framing."
        ),
        when_to_use=[
            "women, girls, wives, feminism, lesbians, or gender roles appear",
            "the dataset/task involves gendered or identity-based harm",
            "retrieved memory mentions misogyny or sexualization",
        ],
        reasoning_steps=[
            "Identify whether women or femininity are the target",
            "Check for objectification, sexual entitlement, gender role policing, or stereotype reinforcement",
            "Distinguish relationship humor from generalized misogyny",
        ],
        positive_indicators=[
            "women are objectified or reduced to sexual/servile roles",
            "women are portrayed as inferior or deserving abuse",
            "anti-feminist framing attacks women as a group",
        ],
        negative_indicators=[
            "individual relationship joke without generalized gender claim",
            "pro-women or anti-misogyny message",
            "gender appears only incidentally",
        ],
        verifier=[
            "Check whether the harmful claim is about women as a class",
            "Do not rely only on sexual words without target framing",
        ],
    ),
    SkillType.HISTORICAL_HARM_TRIVIALIZATION: SkillCard(
        skill_type=SkillType.HISTORICAL_HARM_TRIVIALIZATION,
        name="Historical Harm Trivialization Skill",
        purpose=(
            "Judge whether historical atrocities, war, disaster, illness, or "
            "collective suffering are trivialized or turned into entertainment."
        ),
        when_to_use=[
            "war, genocide, terrorism, slavery, pandemic, illness, death, or disasters appear",
            "retrieved risk memory mentions trivialization",
        ],
        reasoning_steps=[
            "Identify the serious harm context",
            "Decide whether the meme educates/critiques, mocks a person through suffering, or trivializes the crisis",
            "Check whether humor normalizes dangerous attitudes",
        ],
        positive_indicators=[
            "suffering is celebrated or minimized",
            "atrocity or disaster is used as a punchline against victims",
            "a named person's illness, diagnosis, vulnerability, or possible death becomes the joke",
            "dangerous health or crisis misinformation is normalized",
            "pandemic severity, infection, diagnosis, death, or response failure becomes entertainment",
            "COVID hoax, fake-news, mask, vaccine, testing, bleach, or disinfectant framing carries the joke",
        ],
        negative_indicators=[
            "memorial, educational, or critical framing",
            "criticizes the perpetrators or harmful belief",
            "no victim-directed mockery or minimization",
            "generic quarantine inconvenience or self-deprecating coping joke",
            "Corona beer, alcohol, 2020-disaster, or abstract virus jokes without a mocked human target or persuasive medical claim",
        ],
        verifier=[
            "Separate dark humor from endorsement by checking target and implication",
            "Identify who is being laughed at",
        ],
    ),
    SkillType.IMAGE_TEXT_INCONGRUITY: SkillCard(
        skill_type=SkillType.IMAGE_TEXT_INCONGRUITY,
        name="Image-Text Incongruity Skill",
        purpose=(
            "Detect whether image and text combine to create an implied harmful "
            "meaning that neither modality states alone."
        ),
        when_to_use=[
            "image-text relation is unclear, contradictory, or ironic",
            "caption relies on visual template",
            "retrieved insight mentions cross-modal mismatch",
        ],
        reasoning_steps=[
            "Describe image-only meaning and text-only meaning",
            "Describe the combined implication",
            "Decide whether the combined implication supports a harmful or harmless claim",
        ],
        positive_indicators=[
            "cross-modal pairing implies protected-group insult",
            "visual target plus text creates dehumanization or objectification",
            "caption reverses image context into harmful mockery",
        ],
        negative_indicators=[
            "mismatch creates benign absurdity only",
            "text and image are aligned with a non-harmful joke",
            "harmful implication remains speculative",
        ],
        verifier=[
            "Incongruity is evidence only if it yields a concrete harm mechanism",
            "Name the implied proposition explicitly",
        ],
    ),
    SkillType.CULTURAL_TEMPLATE_INTERPRETATION: SkillCard(
        skill_type=SkillType.CULTURAL_TEMPLATE_INTERPRETATION,
        name="Cultural Template Interpretation Skill",
        purpose=(
            "Interpret meme templates, public figures, scenes, slang, and cultural "
            "references that change pragmatic meaning."
        ),
        when_to_use=[
            "celebrity, movie scene, recognizable template, slang, or dog whistle appears",
            "retrieved cases share cultural references",
        ],
        reasoning_steps=[
            "Identify cultural entities or template if visible",
            "Explain typical pragmatic use of the template",
            "Check whether the current caption uses that meaning to harm or critique",
        ],
        positive_indicators=[
            "template meaning reinforces stereotype or dog whistle",
            "cultural reference mocks a vulnerable group",
            "slang encodes hateful or dangerous meaning",
        ],
        negative_indicators=[
            "template is used for generic reaction humor",
            "cultural reference critiques the harmful idea",
            "reference is too uncertain to carry the decision",
        ],
        verifier=[
            "Do not hallucinate a template name if visual evidence is weak",
            "Mark uncertainty when cultural identification is low confidence",
        ],
    ),
    SkillType.COUNTER_SPEECH_QUOTED_CRITICISM: SkillCard(
        skill_type=SkillType.COUNTER_SPEECH_QUOTED_CRITICISM,
        name="Counter-speech / Quoted Criticism Skill",
        purpose=(
            "Distinguish promotion of harmful expression from quoting or mocking "
            "it to criticize the harmful view."
        ),
        when_to_use=[
            "harmful language may be quoted",
            "irony or sarcasm direction is unclear",
            "meme may mock bigotry rather than a protected group",
        ],
        reasoning_steps=[
            "Identify the speaker stance toward the harmful expression",
            "Check whether the meme asks the audience to endorse or reject the harmful view",
            "For HarM, check whether the debunking/quotation still humiliates a named person or makes the dangerous phrase memorable as a joke",
            "Look for quotation, parody, absurdity, or explicit condemnation cues",
        ],
        positive_indicators=[
            "harmful expression is presented as acceptable or funny at victim expense",
            "audience is invited to share the harmful stance",
            "no cues of criticism or distancing",
            "counter-speech format still frames a named person as an idiot, failed leader, diseased target, or deserved victim",
            "quoted hoax, fake-news, disinfectant, mask, vaccine, testing, or COVID-denial language remains the meme's punchline",
            "satire repeats dangerous public-health framing more strongly than it debunks it",
        ],
        negative_indicators=[
            "harmful view is mocked, exposed, or condemned",
            "target is the bigoted speaker or ideology",
            "quotation is clearly critical",
            "dangerous public-health claim is clearly debunked or warned against rather than made entertaining",
        ],
        verifier=[
            "Identify who the joke is on",
            "Do not classify quoted harmful words without stance analysis",
        ],
    ),
}


class SkillRepository:
    """Load and persist skill cards refined from memory."""

    def __init__(self, dataset_name: str, skill_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.skill_dir = skill_dir or os.path.join(
            "data",
            canonical_data_name(dataset_name),
            "skill_memory",
        )
        self.skill_path = os.path.join(self.skill_dir, "skill_cards.json")
        self.skills: Dict[SkillType, SkillCard] = {}

    def load(self, build_if_missing: bool = True) -> Dict[SkillType, SkillCard]:
        os.makedirs(self.skill_dir, exist_ok=True)
        if os.path.exists(self.skill_path):
            with open(self.skill_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.skills = {
                SkillType(item["skill_type"]): SkillCard.from_dict(item)
                for item in raw
            }
        elif build_if_missing:
            self.skills = {key: card for key, card in DEFAULT_SKILL_CARDS.items()}
            self.save()
        return self.skills

    def save(self) -> None:
        os.makedirs(self.skill_dir, exist_ok=True)
        with open(self.skill_path, "w", encoding="utf-8") as f:
            json.dump(
                [card.to_dict() for card in self.skills.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def get(self, skill_type: SkillType) -> SkillCard:
        if not self.skills:
            self.load()
        return self.skills.get(skill_type, DEFAULT_SKILL_CARDS[skill_type])

    def all(self) -> List[SkillCard]:
        if not self.skills:
            self.load()
        return list(self.skills.values())

    def apply_reliability_updates(self, updates: Dict[str, float]) -> None:
        if not updates:
            return
        if not self.skills:
            self.load()
        for key, delta in updates.items():
            try:
                skill_type = SkillType(key)
            except ValueError:
                continue
            card = self.skills.get(skill_type)
            if card:
                card.reliability = clamp_float(card.reliability + delta, card.reliability)
        self.save()


class SkillInducer:
    """Refine seed skill cards from case, insight, and risk memory."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        enabled: bool = False,
        dataset_name: str = "FHM",
    ):
        self.enabled = enabled
        self.boundary_rules = get_harm_boundary_rules(dataset_name)
        self.caller = LLMCaller(client=client, model=model, component_name="SkillInducer")

    def induce(
        self,
        seed_cards: Iterable[SkillCard],
        memory_context: str,
    ) -> List[SkillCard]:
        if not self.enabled:
            return list(seed_cards)

        refined = []
        for card in seed_cards:
            prompt = f'''Refine this reasoning skill card for harmful meme detection.

Memory context:
{memory_context}

Boundary rules that every refined skill must preserve:
{self.boundary_rules}

Seed skill:
{card.to_prompt_block()}

Return JSON only with the same fields:
{{
  "purpose": "...",
  "when_to_use": ["..."],
  "reasoning_steps": ["..."],
  "positive_indicators": ["..."],
  "negative_indicators": ["..."],
  "verifier": ["..."],
  "reliability": <0.0-1.0>
}}

Keep it executable as a reasoning skill. Do not make it a final classifier.'''
            data = extract_json_object(self.caller.call(prompt, max_tokens=1200))
            if data:
                refined.append(
                    SkillCard(
                        skill_type=card.skill_type,
                        name=card.name,
                        purpose=str(data.get("purpose", card.purpose)),
                        when_to_use=_as_list(data.get("when_to_use", card.when_to_use)),
                        reasoning_steps=_as_list(
                            data.get("reasoning_steps", card.reasoning_steps)
                        ),
                        positive_indicators=_as_list(
                            data.get("positive_indicators", card.positive_indicators)
                        ),
                        negative_indicators=_as_list(
                            data.get("negative_indicators", card.negative_indicators)
                        ),
                        verifier=_as_list(data.get("verifier", card.verifier)),
                        reliability=clamp_float(
                            data.get("reliability", card.reliability),
                            card.reliability,
                        ),
                        source_memory_ids=card.source_memory_ids,
                    )
                )
            else:
                refined.append(card)
        return refined


class SkillValidator:
    """Lightweight checks for generated skill cards and evidence packets."""

    @staticmethod
    def validate_card(card: SkillCard) -> bool:
        required = [
            card.purpose,
            card.when_to_use,
            card.reasoning_steps,
            card.positive_indicators,
            card.negative_indicators,
            card.verifier,
        ]
        return all(bool(value) for value in required)

    @staticmethod
    def validate_packet(packet: EvidencePacket) -> EvidencePacket:
        valid_claims = {"Claim_Harmful", "Claim_Harmless"}
        packet.supports = [claim for claim in packet.supports if claim in valid_claims]
        packet.weakens = [claim for claim in packet.weakens if claim in valid_claims]
        if not packet.supports and not packet.weakens:
            packet.uncertainty.append("skill did not produce a valid claim relation")
        packet.confidence = clamp_float(packet.confidence, 0.5)
        return packet


class ReasoningSkillExecutor:
    """Execute selected skills and emit evidence packets."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        use_llm: bool = True,
        dataset_name: str = "FHM",
    ):
        self.use_llm = use_llm
        self.boundary_rules = get_harm_boundary_rules(dataset_name)
        self.dataset_calibration_note = (
            "For FHM, do not over-correct toward harmless. This skill should support "
            "Claim_Harmful when its own perspective can ground an implicit offensive "
            "image-text mechanism: protected-group ridicule, misogyny, women-as-object/"
            "domestic-appliance framing, sexual objectification, rape/consent jokes, "
            "trans or disability mockery, racialized color/food/animal wording, human-"
            "animal comparison aimed at people, religious/nationality stereotyping, "
            "victim mockery, violence normalization, terrorism/atrocity/tragedy humor, "
            "or coded euphemisms such as dishwasher/kitchen/sandwich, vegetable, ape/"
            "monkey/goat/pig, tampon/body jokes, or stop-resisting police-violence "
            "jokes. Still support Claim_Harmless for ordinary slapstick, pets or "
            "standalone animals, family/ex-partner jokes, generic arrest jokes, gross "
            "food jokes, identity/religion/nationality puns, historical descriptions, "
            "anti-hate or perpetrator-directed jokes, and pure absurdity when no such "
            "mechanism is grounded. For close FHM cases, compare the specific pair briefly: "
            "pun vs contempt, object/food wordplay vs human dehumanization, historical "
            "reference vs victim mockery, generic sex joke vs gendered sexual degradation, "
            "and standalone animal/pet joke vs human-animal comparison; do not require "
            "perfect explicitness when the offensive implication is clear."
            if dataset_name.upper() == "FHM" else (
                "For HarM, do not default to Claim_Harmless just because the meme "
                "is political satire, public-figure criticism, or counter-speech. "
                "This skill should support Claim_Harmful when its own perspective "
                "finds weak but concrete somewhat-harmful evidence: a named person, "
                "organization, community, or society is made to look stupid, reckless, "
                "corrupt, unfit, diseased, or deserving ridicule; illness/diagnosis/"
                "death/case-count humor is the punchline; hoax/fake-news/QAnon, "
                "mask/vaccine/testing, disinfectant/bleach, or cure rhetoric becomes "
                "memorable; or disease-origin stigma targets a country/community. "
                "Do not require explicit endorsement or hate speech for HarM. Still "
                "support Claim_Harmless for ordinary policy criticism, abstract virus "
                "jokes, quarantine coping, Corona/alcohol puns, and pure wordplay "
                "when no personal degradation or persuasive risk claim is present."
                if dataset_name.upper() == "HARM" else ""
            )
        )
        self.caller = LLMCaller(client=client, model=model, component_name="Skill")

    def execute(
        self,
        skill_card: SkillCard,
        image_path: str,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
    ) -> EvidencePacket:
        if self.use_llm:
            packet = self._execute_with_llm(
                skill_card, image_path, text, query_signature, memory_result
            )
            if packet:
                return SkillValidator.validate_packet(packet)
            raise LLMCallError(f"Skill {skill_card.key} did not return valid JSON.")
        return SkillValidator.validate_packet(
            self._fallback_execute(skill_card, text, query_signature, memory_result)
        )

    def execute_many(
        self,
        skill_cards: List[SkillCard],
        image_path: str,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> List[EvidencePacket]:
        if not parallel:
            return [
                self.execute(card, image_path, text, query_signature, memory_result)
                for card in skill_cards
            ]

        packets: List[EvidencePacket] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_card = {
                executor.submit(
                    self.execute,
                    card,
                    image_path,
                    text,
                    query_signature,
                    memory_result,
                ): card
                for card in skill_cards
            }
            for future in as_completed(future_to_card):
                try:
                    packets.append(future.result())
                except Exception as exc:
                    card = future_to_card[future]
                    raise LLMCallError(f"Skill {card.key} failed: {exc}") from exc
        return packets

    def _execute_with_llm(
        self,
        skill_card: SkillCard,
        image_path: str,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
    ) -> Optional[EvidencePacket]:
        prompt = f'''Execute the following reasoning skill for harmful meme analysis.

Target meme text: "{text}"
[Image attached if available]

Query signature:
{json.dumps(query_signature.to_dict(), ensure_ascii=False, indent=2)}

Retrieved memory:
{memory_result.get_context_string(top_k_cases=3, top_k_insights=3, top_k_risks=3)}

Boundary rules:
{self.boundary_rules}

Skill card:
{skill_card.to_prompt_block()}

Your task:
- Apply ONLY this skill's perspective.
- Produce an evidence packet, not a final label.
- Use Claim_Harmful and Claim_Harmless exactly for supports/weakens.
- A skill may support Claim_Harmful only when it can name the target, the harmful stance/implication, and concrete evidence for endorsement, ridicule, dehumanization, objectification, stereotyping, threat, or normalization.
- In FHM, the harmful stance can be implicit in the image-text combination. Do not require an explicit slur or threat if this skill can ground a protected-identity, gendered, sexual, violent, dehumanizing, consent-related, or victim-mocking mechanism. Watch for coded/euphemistic meme wording such as dishwasher/kitchen/sandwich, vegetable, ape/monkey/goat/pig, tampon/body jokes, terrorism/atrocity references, or police-violence phrases when the image grounds the target.
- In HarM, a harmful stance/implication can be weak: political cynicism, personal degradation of a named actor, illness/diagnosis/death/case-count humor, hoax/fake-news/QAnon framing, mask/vaccine/testing dismissal, disinfectant/bleach or cure jokes, disease-origin stigma, or pandemic-response failure made entertaining can support Claim_Harmful even without explicit endorsement.
- If this skill only finds sensitive-topic mention, historical description, criticism/denunciation of harm, or ambiguous association, it should support Claim_Harmless or weaken Claim_Harmful.
- For FHM, before supporting Claim_Harmless because the meme is absurd, dark, or ambiguous, explicitly check whether the punchline still degrades a protected group/person, women as a class, a disabled/trans target, a victim, or a person through animalization, sexual objectification, violence, or serious suffering.
- For FHM, before supporting Claim_Harmful from a weak signal, explicitly rule out benign identity/religion/nationality puns, historical/documentary captions, anti-hate commentary, jokes aimed at perpetrators/hate groups/terrorists rather than protected victims, standalone object/appliance/food puns, and non-targeted crude humor.
- For FHM near-neighbor cases, briefly state the image-text anchor if this skill supports Claim_Harmful from coded wording. If the anchor is weak, reduce confidence rather than automatically flipping to Claim_Harmless.
- For HarM, before supporting Claim_Harmless because of satire or counter-speech, explicitly check whether the joke still makes a named person/organization/community the degraded target. Before supporting Claim_Harmful because of COVID keywords, explicitly check whether the case is only abstract virus wordplay, Corona/alcohol humor, quarantine coping, neutral policy accountability, or non-persuasive absurdity.
- Do not use broad words like "trivialization", "stereotype", "objectification", or "dehumanization" unless this skill can point to the exact image-text evidence that makes the target degraded rather than merely joked about.
- Do not let retrieved harmful-looking cases override the target meme's stance; memory is analogy, not a label.

{self.dataset_calibration_note}

Return JSON only:
{{
  "local_claim": "<one sentence local conclusion from this skill>",
  "supports": ["Claim_Harmful" or "Claim_Harmless"],
  "weakens": ["Claim_Harmful" or "Claim_Harmless"],
  "evidence": ["<specific visual/textual/memory evidence>"],
  "uncertainty": ["<missing/ambiguous information>"],
  "confidence": <0.0-1.0>,
  "memory_support": ["<memory ids that support this evidence>"]
}}'''
        response = self.caller.call(
            prompt,
            image_path=image_path if os.path.exists(image_path) else None,
            max_tokens=1200,
        )
        data = extract_json_object(response)
        if not data:
            return None
        return EvidencePacket(
            skill=skill_card.key,
            local_claim=str(data.get("local_claim", "")),
            supports=_as_list(data.get("supports")),
            weakens=_as_list(data.get("weakens")),
            evidence=_as_list(data.get("evidence")),
            uncertainty=_as_list(data.get("uncertainty")),
            confidence=clamp_float(data.get("confidence", 0.5), 0.5),
            memory_support=_as_list(data.get("memory_support")),
            raw=data,
        )

    @staticmethod
    def _fallback_execute(
        skill_card: SkillCard,
        text: str,
        query_signature: QuerySignature,
        memory_result: MemoryRetrievalResult,
    ) -> EvidencePacket:
        lower = (text or "").lower()
        mechanisms = {m.lower() for m in query_signature.possible_harm_mechanisms}
        targets = {t.lower() for t in query_signature.target_candidates}
        supports: List[str] = []
        weakens: List[str] = []
        evidence: List[str] = []
        uncertainty = list(query_signature.uncertainty[:2])

        key = skill_card.skill_type
        if key == SkillType.TARGET_BOUNDARY:
            if targets and (
                "protected_group" in mechanisms
                or "misogyny" in mechanisms
                or "violence" in mechanisms
            ):
                supports.append("Claim_Harmful")
                evidence.append(f"Possible target candidates: {', '.join(sorted(targets)[:5])}.")
            elif any(t in targets for t in ["public figure", "government", "institution"]):
                supports.append("Claim_Harmless")
                weakens.append("Claim_Harmful")
                evidence.append("The target appears closer to public criticism.")
            else:
                weakens.append("Claim_Harmful")
                evidence.append("No explicit vulnerable target is available from the signature.")

        elif key == SkillType.POLITICAL_CRITICISM_BOUNDARY:
            political = "political" in mechanisms or any(
                word in lower for word in ["trump", "biden", "president", "government"]
            )
            dangerous = any(m in mechanisms for m in ["violence", "health_or_disaster"])
            if political and dangerous:
                supports.append("Claim_Harmful")
                evidence.append("Political framing overlaps with violence, crisis, or illness cues.")
            elif political:
                supports.append("Claim_Harmless")
                weakens.append("Claim_Harmful")
                evidence.append("Political target appears to be criticism or satire without clear dangerous cue.")
            else:
                uncertainty.append("No strong political signal for this skill.")

        elif key == SkillType.PROTECTED_GROUP_ATTRIBUTION:
            if "protected_group" in mechanisms:
                supports.append("Claim_Harmful")
                evidence.append("Protected-group terms appear in the signature.")
            else:
                weakens.append("Claim_Harmful")
                evidence.append("No protected-group attribution is detected in fallback analysis.")

        elif key == SkillType.MISOGYNISTIC_FRAMING:
            if "misogyny" in mechanisms or any(
                word in lower for word in ["women", "woman", "girl", "girls", "lesbian"]
            ):
                supports.append("Claim_Harmful")
                evidence.append("Gendered or woman-targeted terms require misogyny scrutiny.")
            else:
                weakens.append("Claim_Harmful")
                evidence.append("No woman-targeted framing is detected.")

        elif key == SkillType.HISTORICAL_HARM_TRIVIALIZATION:
            if any(m in mechanisms for m in ["historical_harm", "health_or_disaster", "violence"]):
                supports.append("Claim_Harmful")
                evidence.append("Serious harm, crisis, or violence cues appear.")
            else:
                weakens.append("Claim_Harmful")
                evidence.append("No serious-harm trivialization cue is detected.")

        elif key == SkillType.IMAGE_TEXT_INCONGRUITY:
            relation = query_signature.image_text_relation.lower()
            if "incongruent" in relation or "contradict" in relation or "image_text_incongruity" in mechanisms:
                if any(m in mechanisms for m in ["protected_group", "misogyny", "violence", "sexualization"]):
                    supports.append("Claim_Harmful")
                    evidence.append("Cross-modal ambiguity overlaps with a plausible harm mechanism.")
                else:
                    supports.append("Claim_Harmless")
                    weakens.append("Claim_Harmful")
                    evidence.append("Incongruity is present but no concrete harm mechanism is explicit.")
            else:
                uncertainty.append("Image-text relation is not clearly incongruent.")

        elif key == SkillType.CULTURAL_TEMPLATE_INTERPRETATION:
            if any(row.item.memory_id for row in memory_result.insight_memories):
                supports.append("Claim_Harmless")
                weakens.append("Claim_Harmful")
                evidence.append("Fallback has no confident cultural-template identification.")
                uncertainty.append("Template interpretation requires visual/cultural confirmation.")
            else:
                uncertainty.append("No cultural memory retrieved.")

        elif key == SkillType.COUNTER_SPEECH_QUOTED_CRITICISM:
            if any(marker in lower for marker in ["not racist", "quote", "\"", "'"]):
                supports.append("Claim_Harmless")
                weakens.append("Claim_Harmful")
                evidence.append("Text contains possible quotation or distancing cue.")
            else:
                uncertainty.append("No clear counter-speech or quotation cue.")

        if not supports and not weakens and evidence:
            weakens.append("Claim_Harmful")

        memory_support = memory_result.get_memory_ids()[:5]
        confidence = 0.58 if supports or weakens else 0.35
        return EvidencePacket(
            skill=skill_card.key,
            local_claim=_fallback_claim(skill_card, supports, weakens),
            supports=supports,
            weakens=weakens,
            evidence=evidence,
            uncertainty=uncertainty,
            confidence=confidence,
            memory_support=memory_support,
            raw={"fallback": True},
        )


def _fallback_claim(
    skill_card: SkillCard,
    supports: List[str],
    weakens: List[str],
) -> str:
    if "Claim_Harmful" in supports:
        return f"{skill_card.name} found local evidence supporting harmful interpretation."
    if "Claim_Harmless" in supports or "Claim_Harmful" in weakens:
        return f"{skill_card.name} found local evidence favoring harmless or weakening harmful interpretation."
    return f"{skill_card.name} found insufficient decisive evidence."


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]
