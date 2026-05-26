# -*- coding: utf-8 -*-
"""Direct one-call multimodal baseline for meme classification."""
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from framework.config import DATASET_CONFIGS, DEFAULT_MODEL, DEFAULT_PATH_CONFIG
from framework.llm_utils import (
    LLMCallError,
    LLMCaller,
    clamp_float,
    extract_json_object,
    normalize_prediction,
)


@dataclass
class DirectBaselineResult:
    """Result from a single direct MLLM classification call."""

    sample_index: int
    image_path: str
    text: str
    actual_label: Optional[int]
    predicted_label: int
    prediction_text: str
    confidence: float
    reasoning: str
    raw_response: str
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.sample_index,
            "image_path": self.image_path,
            "text": self.text,
            "actual": self.actual_label,
            "predicted": self.predicted_label,
            "prediction_text": self.prediction_text,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "processing_time": self.processing_time,
            "mode": "direct",
        }

    def is_correct(self) -> Optional[bool]:
        if self.actual_label is None:
            return None
        return self.predicted_label == self.actual_label


class DirectMemeBaseline:
    """
    Direct baseline: one prompt and one MLLM classification per sample.

    This path does not use query signatures, skill planning, skill execution,
    evidence graphs, memory retrieval, or memory updates.
    """

    def __init__(self, dataset_name: str, model: str = DEFAULT_MODEL):
        self.dataset_name = dataset_name
        self.model = model
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        self.path_config = DEFAULT_PATH_CONFIG
        self.base_path = self.path_config.get_dataset_path(dataset_name)
        self.image_base_path = self.path_config.get_image_path(dataset_name)
        self.caller = LLMCaller(model=model, component_name="DirectBaseline")

    def process_single(
        self,
        image_path: str,
        text: str,
        sample_index: int = 0,
        actual_label: Optional[int] = None,
    ) -> DirectBaselineResult:
        start_time = time.time()
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")
        prompt = self._build_prompt(text=text)

        raw_response = self.caller.call(
            prompt,
            image_path=image_path if os.path.exists(image_path) else None,
            temperature=0.0,
            max_tokens=600,
        )
        data = extract_json_object(raw_response)
        if not data or "prediction" not in data:
            raise LLMCallError("DirectBaseline did not return valid JSON.")

        prediction_text = str(data.get("prediction", "") or "").strip()
        normalized = normalize_prediction(
            prediction_text,
            positive_label=positive_label,
        )
        if not self._is_allowed_prediction(prediction_text, positive_label, negative_label):
            raise LLMCallError(
                "DirectBaseline JSON used an invalid prediction label: "
                f"{prediction_text!r}."
            )

        return DirectBaselineResult(
            sample_index=sample_index,
            image_path=image_path,
            text=text,
            actual_label=actual_label,
            predicted_label=normalized,
            prediction_text=prediction_text,
            confidence=clamp_float(data.get("confidence"), default=0.5),
            reasoning=str(data.get("reasoning", "") or "").strip(),
            raw_response=raw_response,
            processing_time=time.time() - start_time,
        )

    def process_dataset(
        self,
        test_jsonl_path: Optional[str] = None,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        start_from: int = 0,
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
            output_path = os.path.join(
                results_dir,
                f"direct_{self.model}_{timestamp}.jsonl",
            )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self._prepare_output_path(
            output_path=output_path,
            resume_output=resume_output,
            overwrite_output=overwrite_output,
        )

        completed_indices = set()
        correct_count = 0
        total_count = 0
        all_actual: List[int] = []
        all_predicted: List[int] = []
        processing_times: List[float] = []
        error_indices = set()
        existing_has_summary = False

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
            error_indices = state["error_indices"]
            existing_has_summary = state["has_summary"]
            if completed_indices:
                print(
                    f"Resuming direct output: found {len(completed_indices)} "
                    f"completed samples in {output_path}"
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
                    f"Direct processing (resume {len(completed_indices)} done)"
                    if completed_indices else "Direct processing"
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
                        f"Warning: Image not found, using text-only direct call: {image_path}"
                    )

                try:
                    result = self.process_single(
                        image_path=image_path,
                        text=text,
                        sample_index=sample_index,
                        actual_label=label,
                    )
                except LLMCallError as exc:
                    print(f"Direct API/JSON error processing sample {sample_index}: {exc}")
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
                        "mode": "direct",
                    }
                    json.dump(error_dict, f_out, ensure_ascii=False)
                    f_out.write("\n")
                    f_out.flush()
                    continue
                except Exception as exc:
                    print(f"Direct error processing sample {sample_index}: {exc}")
                    continue

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
            processing_times=processing_times,
            output_path=output_path,
            unresolved_error_count=len(error_indices),
            unresolved_error_indices=sorted(error_indices),
        )

        if pending_data or not existing_has_summary:
            with open(output_path, "a", encoding="utf-8") as f_out:
                json.dump({"summary": summary}, f_out, ensure_ascii=False)
                f_out.write("\n")

        print(f"\n=== Direct Results for {self.dataset_name} ===")
        print(f"Accuracy: {summary['accuracy']:.4f} ({correct_count}/{total_count})")
        print(f"Macro F1: {summary['macro_f1']:.4f}")
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"Results saved to: {output_path}")
        return summary

    def _build_prompt(self, text: str) -> str:
        positive_label = self.dataset_config.get("positive_label", "harmful")
        negative_label = self.dataset_config.get("negative_label", "harmless")
        definition = self._dataset_definition(positive_label, negative_label)
        return f'''You are a multimodal meme safety classifier.

Classify the meme using only the attached image and the provided meme text.
Do not use retrieved examples, memory, tools, graphs, previous runs, or hidden labels.

Dataset: {self.dataset_name}
Meme text: "{text}"

Label definitions:
{definition}

Decision guidance:
- Judge the image-text combination, not the text alone.
- Harm can be implicit: the joke may depend on the picture, target, stereotype,
  comparison, role assignment, threat, sexual degradation, or victim mockery.
- Do not over-label ordinary wordplay, absurdity, generic dark humor, neutral
  references, documentary/anti-hate commentary, or jokes without a grounded
  harmful target.
- If the evidence is genuinely insufficient after looking at both image and
  text, choose the less severe label.

Return exactly one JSON object and no extra text:
{{
  "prediction": "{positive_label}" or "{negative_label}",
  "confidence": 0.0,
  "reasoning": "brief image-text reason"
}}'''

    def _dataset_definition(self, positive_label: str, negative_label: str) -> str:
        name = self.dataset_name.upper()
        if positive_label.lower() == "misogynistic":
            return (
                f'- "{positive_label}": the meme attacks, demeans, stereotypes, '
                "objectifies, sexually humiliates, threatens, or assigns inferior "
                "roles to women/girls/feminists because of gender.\n"
                f'- "{negative_label}": the meme does not contain a grounded '
                "misogynistic mechanism, even if it is crude, sexual, ironic, or "
                "generally offensive."
            )
        if name == "HARM":
            return (
                f'- "{positive_label}": the meme endorses, normalizes, trivializes, '
                "or mockingly targets people/groups through hate, violence, "
                "dangerous misinformation, abuse, discrimination, or serious harm.\n"
                f'- "{negative_label}": the meme is neutral, critical of harm, '
                "anti-hate, ordinary satire, or lacks a grounded target/mechanism "
                "of harm."
            )
        return (
            f'- "{positive_label}": the meme attacks, demeans, stereotypes, '
            "dehumanizes, threatens, sexually degrades, objectifies, humiliates, "
            "or mocks a protected/social group or vulnerable victim through the "
            "image-text combination.\n"
            f'- "{negative_label}": the meme lacks a grounded harmful mechanism, '
            "even if it contains edgy wording, political or historical references, "
            "ordinary insults, absurdity, or non-targeted dark humor."
        )

    @staticmethod
    def _is_allowed_prediction(
        prediction_text: str,
        positive_label: str,
        negative_label: str,
    ) -> bool:
        normalized = prediction_text.strip().lower()
        return normalized in {
            positive_label.lower(),
            negative_label.lower(),
            "harmful",
            "harmless",
            "misogynistic",
            "not misogynistic",
        }

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

    @staticmethod
    def _load_existing_output_state(
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
            "error_indices": set(),
            "has_summary": False,
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
                try:
                    sample_index = int(row.get("index"))
                except (TypeError, ValueError):
                    continue
                if valid_indices is not None and sample_index not in valid_indices:
                    continue
                if row.get("status") == "error" or row.get("error"):
                    state["error_indices"].add(sample_index)
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

    def _summarize_results(
        self,
        all_actual: List[int],
        all_predicted: List[int],
        total_count: int,
        correct_count: int,
        processing_times: List[float],
        output_path: str,
        unresolved_error_count: int = 0,
        unresolved_error_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        accuracy = correct_count / total_count if total_count else 0.0
        macro_f1, precision, recall = self._classification_metrics(
            all_actual,
            all_predicted,
        )
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        return {
            "dataset": self.dataset_name,
            "model": self.model,
            "mode": "direct",
            "total_samples": total_count,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "average_processing_time": avg_time,
            "unresolved_error_count": unresolved_error_count,
            "unresolved_error_indices": unresolved_error_indices or [],
            "output_path": output_path,
        }

    @staticmethod
    def _classification_metrics(
        all_actual: List[int],
        all_predicted: List[int],
    ) -> Tuple[float, float, float]:
        if not all_actual:
            return 0.0, 0.0, 0.0
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score

            return (
                f1_score(all_actual, all_predicted, average="macro", zero_division=0),
                precision_score(all_actual, all_predicted, average="macro", zero_division=0),
                recall_score(all_actual, all_predicted, average="macro", zero_division=0),
            )
        except Exception:
            labels = sorted(set(all_actual) | set(all_predicted))
            precisions = []
            recalls = []
            f1s = []
            for label in labels:
                tp = sum(
                    1 for actual, pred in zip(all_actual, all_predicted)
                    if actual == label and pred == label
                )
                fp = sum(
                    1 for actual, pred in zip(all_actual, all_predicted)
                    if actual != label and pred == label
                )
                fn = sum(
                    1 for actual, pred in zip(all_actual, all_predicted)
                    if actual == label and pred != label
                )
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                )
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            return (
                sum(f1s) / len(f1s),
                sum(precisions) / len(precisions),
                sum(recalls) / len(recalls),
            )
