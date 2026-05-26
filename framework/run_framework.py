# -*- coding: utf-8 -*-
"""
Entry point for the skill-memory harmful meme framework.
"""
import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    FrameworkConfig,
    canonical_data_name,
)
from framework.direct_baseline import DirectMemeBaseline
from framework.memory import SkillMemoryStore
from framework.pipeline import AblationPipeline, MemeDetectionPipeline
from framework.skills import SkillType


def build_skill_memory(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    use_llm: bool = False,
    max_cases: Optional[int] = None,
    resume: bool = True,
    memory_dir: Optional[str] = None,
) -> Dict:
    print(f"\n{'=' * 60}")
    print("Building Label-free Skill Memory and Skill Cards")
    print(f"Dataset: {dataset_name}")
    print(f"Use MLLM annotator: {use_llm}")
    print(f"Resume checkpoint: {resume}")
    print(f"{'=' * 60}\n")

    pipeline = MemeDetectionPipeline(
        dataset_name=dataset_name,
        model=model,
        memory_dir=memory_dir,
        skill_dir=memory_dir,
        use_memory=True,
        preload_memory=False,
        use_llm_signature=False,
        use_llm_planner=False,
        use_llm_skills=False,
        use_llm_judge=False,
    )
    stats = pipeline.build_memory(use_llm=use_llm, max_cases=max_cases, resume=resume)
    print(f"Skill memory and skill cards ready: {stats}")
    return stats


def run_main_experiment(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    output_path: Optional[str] = None,
    use_memory: bool = True,
    top_k: int = 5,
    enable_memory_update: bool = True,
    deterministic: bool = False,
    use_surface_retrieval: bool = True,
    surface_retrieval_backend: str = "auto",
    surface_ssr_path: Optional[str] = None,
    resume_output: bool = True,
    isolate_memory_updates: bool = True,
    run_memory_dir: Optional[str] = None,
    base_memory_dir: Optional[str] = None,
    parallel_skills: bool = True,
    max_workers: int = 1,
    memory_update_policy: str = "proposal_only",
    overwrite_output: bool = False,
) -> Dict:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"skill_memory_main_{model}_{timestamp}.jsonl",
        )

    active_memory_dir = None
    if use_memory and enable_memory_update and isolate_memory_updates:
        active_memory_dir = prepare_run_memory_dir(
            dataset_name=dataset_name,
            output_path=output_path,
            run_memory_dir=run_memory_dir,
            base_memory_dir=base_memory_dir,
        )
    elif use_memory and enable_memory_update and not isolate_memory_updates:
        data_dir_name = canonical_data_name(dataset_name)
        print(
            "Warning: online memory update will write into the base "
            f"data/{data_dir_name}/skill_memory directory."
        )
    elif use_memory and not enable_memory_update:
        active_memory_dir = resolve_base_memory_dir(
            dataset_name=dataset_name,
            base_memory_dir=base_memory_dir,
            warn_on_fallback=False,
        )

    print(f"\n{'=' * 60}")
    print("Running Skill-Memory Main Experiment")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Use Memory: {use_memory}")
    print(f"Top-K Memory: {top_k}")
    print(f"Surface Retrieval: {use_surface_retrieval} ({surface_retrieval_backend})")
    print(f"Memory Update: {enable_memory_update}")
    print(f"Memory Update Policy: {memory_update_policy}")
    print(f"Memory Update Isolation: {bool(active_memory_dir)}")
    if active_memory_dir:
        print(f"Run Memory Dir: {active_memory_dir}")
    print(f"Parallel Skills: {parallel_skills}")
    print(f"Max Workers: {max_workers}")
    print(f"Resume Output: {resume_output}")
    print(f"Deterministic fallback: {deterministic}")
    print(f"{'=' * 60}\n")

    config = FrameworkConfig(
        top_k_retrieval=top_k,
        use_surface_retrieval=use_surface_retrieval,
        surface_retrieval_backend=surface_retrieval_backend,
        surface_ssr_path=surface_ssr_path,
        parallel_tool_execution=parallel_skills,
        max_workers=max_workers,
        memory_update_policy=memory_update_policy,
        llm_model=model,
        vision_model=model,
    )
    pipeline = MemeDetectionPipeline(
        dataset_name=dataset_name,
        config=config,
        model=model,
        use_memory=use_memory,
        preload_memory=use_memory,
        use_llm_signature=not deterministic,
        use_llm_planner=not deterministic,
        use_llm_skills=not deterministic,
        use_llm_judge=not deterministic,
        enable_memory_update=enable_memory_update,
        memory_dir=active_memory_dir,
        skill_dir=active_memory_dir,
    )

    return pipeline.process_dataset(
        max_samples=max_samples,
        output_path=output_path,
        use_memory=use_memory,
        resume_output=resume_output,
        overwrite_output=overwrite_output,
    )


def prepare_run_memory_dir(
    dataset_name: str,
    output_path: str,
    run_memory_dir: Optional[str] = None,
    base_memory_dir: Optional[str] = None,
) -> str:
    """Create or reuse a per-run memory copy for online updates."""
    base_memory_dir = resolve_base_memory_dir(
        dataset_name=dataset_name,
        base_memory_dir=base_memory_dir,
        warn_on_fallback=True,
    )

    if run_memory_dir is None:
        output_dir = os.path.dirname(output_path) or os.path.join("results", dataset_name)
        output_stem = os.path.splitext(os.path.basename(output_path))[0]
        run_memory_dir = os.path.join(output_dir, "run_memory", output_stem)

    if os.path.isdir(run_memory_dir) and os.listdir(run_memory_dir):
        print(f"Reusing existing run memory copy: {run_memory_dir}")
        return run_memory_dir

    os.makedirs(os.path.dirname(run_memory_dir), exist_ok=True)
    shutil.copytree(base_memory_dir, run_memory_dir, dirs_exist_ok=True)
    print(f"Copied base skill memory to run memory: {run_memory_dir}")
    return run_memory_dir


def resolve_base_memory_dir(
    dataset_name: str,
    base_memory_dir: Optional[str] = None,
    warn_on_fallback: bool = True,
) -> Optional[str]:
    """Resolve the read-only base memory directory for evaluation runs."""
    data_dir_name = canonical_data_name(dataset_name)
    preferred_dir = base_memory_dir or os.path.join("data", data_dir_name, "skill_memory_base")
    if os.path.isdir(preferred_dir):
        return preferred_dir

    fallback_dir = os.path.join("data", data_dir_name, "skill_memory")
    if os.path.isdir(fallback_dir):
        if warn_on_fallback:
            print(
                f"Warning: clean base memory not found: {preferred_dir}. "
                f"Falling back to {fallback_dir}."
            )
        return fallback_dir

    if base_memory_dir:
        raise FileNotFoundError(f"Base skill memory directory not found: {base_memory_dir}")
    return None


def freeze_base_memory(
    dataset_name: str,
    source_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    reset_skill_reliability: bool = True,
) -> str:
    """Create a clean immutable-ish base memory snapshot for future runs."""
    data_dir_name = canonical_data_name(dataset_name)
    source_dir = source_dir or os.path.join("data", data_dir_name, "skill_memory")
    target_dir = target_dir or os.path.join("data", data_dir_name, "skill_memory_base")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source memory directory not found: {source_dir}")

    if os.path.isdir(target_dir):
        backup_dir = target_dir + ".bak"
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(target_dir, backup_dir)
        print(f"Existing base memory moved to backup: {backup_dir}")

    shutil.copytree(source_dir, target_dir)

    risk_path = os.path.join(target_dir, "risk_memory.jsonl")
    if os.path.exists(risk_path):
        with open(risk_path, "r", encoding="utf-8") as f:
            risks = [json.loads(line) for line in f if line.strip()]
        base_risks = [
            row for row in risks
            if not str(row.get("memory_id", "")).startswith(f"risk_{dataset_name}_online_")
        ][:6]
        with open(risk_path, "w", encoding="utf-8") as f:
            for row in base_risks:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

    skill_stats_path = os.path.join(target_dir, "skill_stats.json")
    with open(skill_stats_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

    if reset_skill_reliability:
        skill_cards_path = os.path.join(target_dir, "skill_cards.json")
        if os.path.exists(skill_cards_path):
            with open(skill_cards_path, "r", encoding="utf-8") as f:
                cards = json.load(f)
            for card in cards:
                card["reliability"] = 0.65
            with open(skill_cards_path, "w", encoding="utf-8") as f:
                json.dump(cards, f, ensure_ascii=False, indent=2)

    print(f"Frozen base memory saved to: {target_dir}")
    return target_dir


def run_ablation_study(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    deterministic: bool = False,
) -> Dict[str, Dict]:
    print(f"\n{'=' * 60}")
    print("Running Skill-Memory Ablation Study")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"{'=' * 60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    common = {
        "dataset_name": dataset_name,
        "model": model,
        "use_llm_signature": not deterministic,
        "use_llm_planner": not deterministic,
        "use_llm_skills": not deterministic,
        "use_llm_judge": not deterministic,
    }
    configs = [
        ("full", {"use_memory": True, "use_planning": True}),
        ("no_memory", {"use_memory": False, "use_planning": True}),
        ("no_planning_default_skills", {"use_memory": True, "use_planning": False}),
        ("all_skills", {"use_memory": True, "use_planning": False, "use_all_skills": True}),
    ]

    results: Dict[str, Dict] = {}
    for name, params in configs:
        print(f"\n--- Ablation: {name} ---")
        pipeline = AblationPipeline(**common, **params)
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"skill_ablation_{name}_{timestamp}.jsonl",
        )
        results[name] = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=output_path,
            use_memory=params.get("use_memory", True),
        )

    summary_path = os.path.join(
        output_dir,
        dataset_name,
        f"skill_ablation_summary_{timestamp}.json",
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def run_retrieval_sensitivity(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    k_values: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    deterministic: bool = False,
) -> Dict[str, Dict]:
    k_values = k_values or [1, 3, 5, 7, 10]
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for k in k_values:
        print(f"\n--- Testing top_k_memory={k} ---")
        config = FrameworkConfig(top_k_retrieval=k)
        pipeline = MemeDetectionPipeline(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_memory=True,
            use_llm_signature=not deterministic,
            use_llm_planner=not deterministic,
            use_llm_skills=not deterministic,
            use_llm_judge=not deterministic,
        )
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"skill_sensitivity_k{k}_{timestamp}.jsonl",
        )
        results[f"k={k}"] = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=output_path,
        )
    return results


def run_skill_count_sensitivity(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    skill_counts: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    deterministic: bool = False,
) -> Dict[str, Dict]:
    skill_counts = skill_counts or [1, 2, 3, 4, 5, 6, 8]
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for count in skill_counts:
        print(f"\n--- Testing n_skills={count} ---")
        config = FrameworkConfig(
            min_skills_to_select=count,
            max_skills_to_select=count,
            fixed_skill_count=count,
            min_tools_to_select=count,
            max_tools_to_select=count,
            fixed_tool_count=count,
        )
        pipeline = MemeDetectionPipeline(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_memory=True,
            use_llm_signature=not deterministic,
            use_llm_planner=not deterministic,
            use_llm_skills=not deterministic,
            use_llm_judge=not deterministic,
        )
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"skill_sensitivity_n{count}_{timestamp}.jsonl",
        )
        results[f"n_skills={count}"] = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=output_path,
        )
    return results


def run_llm_robustness(
    dataset_name: str,
    models: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
) -> Dict[str, Dict]:
    models = models or ["gemini-flash", "gpt-4o-mini", "gpt-4o", "qwen-plus"]
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model in models:
        if model not in AVAILABLE_MODELS:
            print(f"Warning: Model {model} is not configured; skipping.")
            continue
        print(f"\n--- Testing model: {model} ---")
        try:
            pipeline = MemeDetectionPipeline(
                dataset_name=dataset_name,
                model=model,
                use_memory=True,
            )
            output_path = os.path.join(
                output_dir,
                dataset_name,
                f"skill_llm_{model}_{timestamp}.jsonl",
            )
            results[model] = pipeline.process_dataset(
                max_samples=max_samples,
                output_path=output_path,
            )
        except Exception as exc:
            print(f"Error with model {model}: {exc}")
    return results


def run_efficiency_analysis(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: int = 50,
    output_dir: str = "results",
    deterministic: bool = False,
) -> Dict[str, Dict]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    for parallel in [True, False]:
        name = "parallel" if parallel else "sequential"
        print(f"\n--- Efficiency: {name} skill execution ---")
        config = FrameworkConfig(parallel_tool_execution=parallel, max_workers=4)
        pipeline = AblationPipeline(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_memory=True,
            use_planning=False,
            use_all_skills=True,
            use_llm_signature=not deterministic,
            use_llm_planner=not deterministic,
            use_llm_skills=not deterministic,
            use_llm_judge=not deterministic,
        )
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"skill_efficiency_{name}_{timestamp}.jsonl",
        )
        results[name] = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=output_path,
        )

    p_time = results["parallel"]["average_processing_time"]
    s_time = results["sequential"]["average_processing_time"]
    results["speedup"] = s_time / p_time if p_time > 0 else 0
    print(f"\nSpeedup: {results['speedup']:.2f}x")
    return results


def run_direct_baseline(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    output_path: Optional[str] = None,
    test_jsonl_path: Optional[str] = None,
    resume_output: bool = True,
    overwrite_output: bool = False,
) -> Dict:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            output_dir,
            dataset_name,
            f"direct_{model}_{timestamp}.jsonl",
        )

    print(f"\n{'=' * 60}")
    print("Running Direct One-Call MLLM Baseline")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print("Use Memory: False")
    print("Use Skills: False")
    print("Use Evidence Graph: False")
    print("Memory Update: False")
    print(f"Resume Output: {resume_output}")
    print(f"{'=' * 60}\n")

    pipeline = DirectMemeBaseline(dataset_name=dataset_name, model=model)
    return pipeline.process_dataset(
        test_jsonl_path=test_jsonl_path,
        output_path=output_path,
        max_samples=max_samples,
        resume_output=resume_output,
        overwrite_output=overwrite_output,
    )


def verify_memory(dataset_name: str) -> bool:
    store = SkillMemoryStore(dataset_name)
    store.load(build_if_missing=True)
    print(f"Skill memory stats: {store.get_statistics()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skill-Memory Framework for Harmful Meme Detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="main",
        choices=[
            "main",
            "direct",
            "build_memory",
            "freeze_base_memory",
            "verify_memory",
            "ablation",
            "retrieval_sensitivity",
            "skill_sensitivity",
            "llm_robustness",
            "efficiency",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FHM",
        choices=["FHM", "HarM", "Harm-C", "Harm-P", "MultiOFF", "PrideMM"],
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default=None,
        help="Optional test JSONL path. Used by --mode direct.",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="JSONL result path. If it exists, --mode main resumes it by default.",
    )
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--disable_surface_retrieval",
        action="store_true",
        help="Disable surface-similar case retrieval and use logic memory only.",
    )
    parser.add_argument(
        "--surface_retrieval_backend",
        type=str,
        default="auto",
        choices=["auto", "ssr", "clip", "text"],
        help="Surface backend when enabled. auto prefers data/{dataset}/{dataset}_SSR.jsonl.",
    )
    parser.add_argument(
        "--surface_ssr_path",
        type=str,
        default=None,
        help="Optional SSR jsonl path. Defaults to data/{dataset}/{dataset}_SSR.jsonl.",
    )
    parser.add_argument(
        "--no_resume_output",
        action="store_true",
        help="Do not skip rows already present in --output_path.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help=(
            "Allow --no_resume_output to replace an existing non-empty output file. "
            "Without this flag, existing outputs are protected from accidental reruns."
        ),
    )
    parser.add_argument(
        "--disable_memory_update",
        action="store_true",
        help="Disable online self-evolving memory and skill reliability updates.",
    )
    parser.add_argument(
        "--memory_update_policy",
        type=str,
        default="proposal_only",
        choices=["active", "no_case", "risk_only", "proposal_only"],
        help=(
            "Control which self-evolve proposals are applied to the live memory. "
            "active applies all accepted proposals; no_case blocks pseudo case writes; "
            "risk_only applies only risk memory; proposal_only records proposals in "
            "the result JSONL without updating memory."
        ),
    )
    parser.add_argument(
        "--update_memory_in_place",
        action="store_true",
        help=(
            "Write online memory updates into data/{dataset}/skill_memory. "
            "By default, main runs update an isolated copy under results/."
        ),
    )
    parser.add_argument(
        "--run_memory_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for the isolated run memory copy. "
            "Defaults to results/{dataset}/run_memory/{output_stem}."
        ),
    )
    parser.add_argument(
        "--base_memory_dir",
        type=str,
        default=None,
        help=(
            "Base memory directory copied for isolated online updates. "
            "Defaults to data/{dataset}/skill_memory_base, falling back to skill_memory."
        ),
    )
    parser.add_argument(
        "--disable_parallel_skills",
        action="store_true",
        help="Run selected skills sequentially instead of concurrent API calls.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum concurrent skill workers when parallel skills are enabled.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic local fallbacks instead of LLM calls.",
    )
    parser.add_argument(
        "--build_with_llm",
        action="store_true",
        help="Use MLLM annotator when building memory.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from case_memory.partial.jsonl during MLLM memory build.",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,3,5,7,10",
        help="Comma-separated top-k values for retrieval sensitivity.",
    )
    parser.add_argument(
        "--skill_counts",
        type=str,
        default="1,2,3,4,5,6,8",
        help="Comma-separated skill counts for sensitivity analysis.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "build_memory":
        build_skill_memory(
            dataset_name=args.dataset,
            model=args.model,
            use_llm=args.build_with_llm,
            max_cases=args.max_samples,
            resume=not args.no_resume,
            memory_dir=args.base_memory_dir,
        )
    elif args.mode == "freeze_base_memory":
        freeze_base_memory(
            dataset_name=args.dataset,
            source_dir=args.base_memory_dir,
        )
    elif args.mode == "verify_memory":
        verify_memory(args.dataset)
    elif args.mode == "main":
        run_main_experiment(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            output_path=args.output_path,
            use_memory=not args.no_memory,
            top_k=args.top_k,
            enable_memory_update=not args.disable_memory_update,
            deterministic=args.deterministic,
            use_surface_retrieval=not args.disable_surface_retrieval,
            surface_retrieval_backend=args.surface_retrieval_backend,
            surface_ssr_path=args.surface_ssr_path,
            resume_output=not args.no_resume_output,
            isolate_memory_updates=not args.update_memory_in_place,
            run_memory_dir=args.run_memory_dir,
            base_memory_dir=args.base_memory_dir,
            parallel_skills=not args.disable_parallel_skills,
            max_workers=max(1, args.max_workers),
            memory_update_policy=args.memory_update_policy,
            overwrite_output=args.overwrite_output,
        )
    elif args.mode == "direct":
        run_direct_baseline(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            output_path=args.output_path,
            test_jsonl_path=args.test_jsonl_path,
            resume_output=not args.no_resume_output,
            overwrite_output=args.overwrite_output,
        )
    elif args.mode == "ablation":
        run_ablation_study(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
        )
    elif args.mode == "retrieval_sensitivity":
        run_retrieval_sensitivity(
            dataset_name=args.dataset,
            model=args.model,
            k_values=[int(v) for v in args.k_values.split(",") if v],
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
        )
    elif args.mode == "skill_sensitivity":
        run_skill_count_sensitivity(
            dataset_name=args.dataset,
            model=args.model,
            skill_counts=[int(v) for v in args.skill_counts.split(",") if v],
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
        )
    elif args.mode == "llm_robustness":
        run_llm_robustness(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    elif args.mode == "efficiency":
        run_efficiency_analysis(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples or 50,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
        )


if __name__ == "__main__":
    main()
