# -*- coding: utf-8 -*-
"""
Configuration for the skill-memory framework for zero-shot harmful meme detection.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ===================== API Configuration =====================
# Set OPENAI_API_KEY in your shell before running LLM-based experiments.
# OPENAI_BASE_URL is optional and can point to any OpenAI-compatible endpoint.
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ===================== Model Configuration =====================
@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    name: str
    api_name: str  # Name used in API calls
    supports_vision: bool = True
    max_tokens: int = 2048
    temperature: float = 0.0
    
# Available models for experiments
AVAILABLE_MODELS = {
    # Primary model aliases. Edit api_name or pass --model to match your endpoint.
    "gemini-flash": ModelConfig(
        name=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash"),
        api_name=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash"),
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    # Alternative models for robustness experiments
    "gpt-4o-mini": ModelConfig(
        name=os.getenv("GPT_4O_MINI_MODEL", "gpt-4o-mini"),
        api_name=os.getenv("GPT_4O_MINI_MODEL", "gpt-4o-mini"),
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    "gpt-5.4-mini": ModelConfig(
        name=os.getenv("GPT_54_MINI_MODEL", "gpt-5.4-mini"),
        api_name=os.getenv("GPT_54_MINI_MODEL", "gpt-5.4-mini"),
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    "gpt-4o": ModelConfig(
        name=os.getenv("GPT_4O_MODEL", "gpt-4o"),
        api_name=os.getenv("GPT_4O_MODEL", "gpt-4o"),
        supports_vision=True,
        max_tokens=4096,
        temperature=0.0
    ),
    "qwen-plus": ModelConfig(
        name="Qwen Plus",
        api_name="qwen-plus",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    "qwen-vl-max": ModelConfig(
        name="Qwen VL Max",
        api_name="qwen-vl-max",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
}

# Default model selection
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ===================== Framework Configuration =====================
@dataclass
class FrameworkConfig:
    """Main configuration for the meme detection framework"""
    # Retrieval settings
    top_k_retrieval: int = 5  # Number of similar samples to retrieve
    retrieval_weight_text: float = 0.2  # Weight for text similarity
    retrieval_weight_image: float = 0.8  # Weight for image similarity
    use_surface_retrieval: bool = True  # Enable surface-similar case retrieval
    surface_retrieval_backend: str = "auto"  # auto, ssr, clip, or text
    surface_ssr_path: Optional[str] = None  # Optional precomputed SSR jsonl path
    surface_top_k: Optional[int] = None  # If None, use top_k_retrieval
    logic_top_k: Optional[int] = None  # If None, use top_k_retrieval
    surface_fusion_weight: float = 0.5
    logic_fusion_weight: float = 0.5
    clip_model_name: str = "clip-ViT-B-32"
    clip_batch_size: int = 32
    
    # Skill planning settings
    max_skills_to_select: int = 4  # Maximum number of skills to select per sample
    min_skills_to_select: int = 2  # Minimum number of skills to select
    fixed_skill_count: Optional[int] = None  # If set, Planner MUST select exactly this many skills

    # Backward-compatible aliases used by some experiment scripts
    max_tools_to_select: int = 4
    min_tools_to_select: int = 2
    fixed_tool_count: Optional[int] = None
    
    # Model settings
    llm_model: str = DEFAULT_MODEL
    vision_model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
    # Execution settings
    parallel_tool_execution: bool = True  # Enable parallel skill execution
    max_workers: int = 1  # Maximum parallel workers; 1 is safer for unstable APIs
    memory_update_policy: str = "proposal_only"  # active, no_case, risk_only, proposal_only
    
    # Output settings
    verbose: bool = True  # Detailed logging
    save_intermediate: bool = True  # Save intermediate results
    

# ===================== Tool Configuration =====================
@dataclass
class ToolConfig:
    """Configuration for cognitive tools"""
    name: str
    description: str
    enabled: bool = True
    requires_vision: bool = False
    priority: int = 5  # 1-10, higher is more important

# Define all 8 cognitive tools
COGNITIVE_TOOLS = {
    "sentiment_reversal": ToolConfig(
        name="Sentiment Reversal Detector",
        description="Analyzes sentiment polarity contrast between text and image",
        requires_vision=True,
        priority=8
    ),
    "image_text_aligner": ToolConfig(
        name="Fine-grained Image-Text Aligner",
        description="Checks entity and attribute consistency between text and image",
        requires_vision=True,
        priority=9
    ),
    "visual_rhetoric": ToolConfig(
        name="Visual Rhetoric Decoder",
        description="Identifies visual rhetorical devices like exaggeration and juxtaposition",
        requires_vision=True,
        priority=6
    ),
    "micro_expression": ToolConfig(
        name="Micro-Expression Analyzer",
        description="Analyzes facial expressions in relation to textual context",
        requires_vision=True,
        priority=7
    ),
    "culture_retriever": ToolConfig(
        name="Culture Knowledge Retriever",
        description="Identifies cultural references, celebrities, and contextual knowledge",
        requires_vision=True,
        priority=8
    ),
    "pragmatic_irony": ToolConfig(
        name="Pragmatic Irony Identifier",
        description="Detects linguistic irony markers like rhetorical questions and sarcasm",
        requires_vision=False,
        priority=7
    ),
    "scene_text_ocr": ToolConfig(
        name="Scene Text OCR Integrator",
        description="Extracts and analyzes embedded text within images",
        requires_vision=True,
        priority=6
    ),
    "target_identifier": ToolConfig(
        name="Target Identification Probe",
        description="Identifies the target of the meme (self-deprecation, individual, social phenomenon)",
        requires_vision=True,
        priority=5
    ),
}


# ===================== Dataset Configuration =====================
DATASET_CONFIGS = {
    "FHM": {
        "image_key": "img",
        "text_key": "text",
        "label_key": "label",
        "label_mapping": None,
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "HarM": {
        "data_dir": "Harm-C",
        "image_key": "image",
        "text_key": "text",
        "label_key": "labels",
        "label_mapping": {"not harmful": 0, "harmless": 0, "somewhat harmful": 1, "very harmful": 1, "default_harmful": 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "Harm-C": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "labels",
        "label_mapping": {"not harmful": 0, "harmless": 0, "somewhat harmful": 1, "very harmful": 1, "default_harmful": 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "Harm-P": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "labels",
        "label_mapping": {"not harmful": 0, "harmless": 0, "somewhat harmful": 1, "very harmful": 1, "default_harmful": 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "MultiOFF": {
        "image_key": "image_name",
        "text_key": "sentence",
        "label_key": "label",
        "label_mapping": {"non-offensive": 0, "non-offensiv": 0, "not offensive": 0, "offensive": 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "PrideMM": {
        "image_key": "name",
        "text_key": "text",
        "label_key": "hate",
        "label_mapping": {"0": 0, "1": 1, 0: 0, 1: 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    }
}


def canonical_data_name(dataset_name: str) -> str:
    """Return the directory name used under data/ for a dataset alias."""
    config = DATASET_CONFIGS.get(dataset_name, {})
    return config.get("data_dir", dataset_name)


# ===================== Path Configuration =====================
@dataclass
class PathConfig:
    """Path configuration for data and results"""
    base_dir: str = "."
    data_dir: str = "data"
    results_dir: str = "results"
    cache_dir: str = "cache"
    embeddings_dir: str = "embeddings"
    knowledge_base_dir: str = "knowledge_base"
    
    def get_dataset_path(self, dataset_name: str) -> str:
        return os.path.join(self.base_dir, self.data_dir, canonical_data_name(dataset_name))
    
    def get_image_path(self, dataset_name: str) -> str:
        return os.path.join(self.get_dataset_path(dataset_name), "images")
    
    def get_results_path(self, dataset_name: str) -> str:
        path = os.path.join(self.base_dir, self.results_dir, dataset_name)
        os.makedirs(path, exist_ok=True)
        return path


# Create default instances
DEFAULT_FRAMEWORK_CONFIG = FrameworkConfig()
DEFAULT_PATH_CONFIG = PathConfig()
