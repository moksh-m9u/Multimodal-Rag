"""Provider and model registry for LLM selection.

Central place to define available providers, their models, and metadata
(pricing, context window, capabilities).  Used by the dashboard dropdown
and the backend /query/custom endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Provider and model definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single model."""
    id: str                    # model identifier used in API calls
    display_name: str          # human-readable name for UI
    context_window: int        # max context tokens
    input_price_per_1m: float  # USD per 1M input tokens (0 if unknown/free)
    output_price_per_1m: float # USD per 1M output tokens (0 if unknown/free)
    supports_vision: bool = False
    supports_streaming: bool = True
    description: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for a provider."""
    id: str                    # short key: "gemini", "groq", "huggingface", "openai"
    display_name: str          # "Google Gemini", "Groq", "Hugging Face", "OpenAI"
    api_key_env: str           # env var name for default key (e.g. "GEMINI_API_KEY")
    base_url: str | None = None # custom base URL (for HF router, etc.)
    models: tuple[ModelInfo, ...] = ()
    description: str = ""


# ---- Google Gemini ----
GEMINI_MODELS = (
    ModelInfo(
        id="gemini-3.7-flash",
        display_name="Gemini 3.7 Flash (New)",
        context_window=1_000_000,
        input_price_per_1m=0.75,
        output_price_per_1m=3.75,
        supports_vision=True,
        description="Newest budget Flash model (promo pricing through Dec 2026)",
    ),
    ModelInfo(
        id="gemini-3.6-flash",
        display_name="Gemini 3.6 Flash",
        context_window=1_000_000,
        input_price_per_1m=1.50,
        output_price_per_1m=7.50,
        supports_vision=True,
        description="Workhorse Flash model with strong multimodal capabilities",
    ),
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        context_window=1_000_000,
        input_price_per_1m=1.50,
        output_price_per_1m=9.00,
        supports_vision=True,
        description="Frontier intelligence with speed and grounding",
    ),
    ModelInfo(
        id="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        context_window=1_000_000,
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
        supports_vision=True,
        description="Most cost-effective 3.5 generation model for high-volume tasks",
    ),
    ModelInfo(
        id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro (Preview)",
        context_window=1_000_000,
        input_price_per_1m=2.00,
        output_price_per_1m=12.00,
        supports_vision=True,
        description="Flagship reasoning model for complex tasks, coding, agentic work",
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        context_window=1_000_000,
        input_price_per_1m=0.25,
        output_price_per_1m=1.50,
        supports_vision=True,
        description="High-volume simple tasks: translation, extraction, agents",
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash (Preview)",
        context_window=1_000_000,
        input_price_per_1m=0.50,
        output_price_per_1m=3.00,
        supports_vision=True,
        description="Fast, capable production default",
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        context_window=1_000_000,
        input_price_per_1m=1.25,
        output_price_per_1m=10.00,
        supports_vision=True,
        description="Mature reasoning & coding model with 1M context",
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        context_window=1_000_000,
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
        supports_vision=True,
        description="Balanced cost/performance for production chatbots",
    ),
    ModelInfo(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        context_window=1_000_000,
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        supports_vision=True,
        description="Cheapest option, ultra low-cost high-throughput",
    ),
)

# ---- Groq (all available models from API) ----
GROQ_MODELS = (
    ModelInfo(
        id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B Versatile",
        context_window=131_072,
        input_price_per_1m=0.59,
        output_price_per_1m=0.79,
        description="280 T/s — flagship open-weight reasoning",
    ),
    ModelInfo(
        id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant",
        context_window=131_072,
        input_price_per_1m=0.05,
        output_price_per_1m=0.08,
        description="560 T/s — fast, cheapest Llama",
    ),
    ModelInfo(
        id="openai/gpt-oss-120b",
        display_name="OpenAI GPT OSS 120B",
        context_window=131_072,
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        description="500 T/s — large open-weight OpenAI model",
    ),
    ModelInfo(
        id="openai/gpt-oss-20b",
        display_name="OpenAI GPT OSS 20B",
        context_window=131_072,
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        description="1000 T/s — fastest OpenAI model on Groq",
    ),
    ModelInfo(
        id="qwen/qwen3.6-27b",
        display_name="Qwen 3.6 27B",
        context_window=131_072,
        input_price_per_1m=0.30,
        output_price_per_1m=0.30,
        description="Multilingual and coding model",
    ),
    ModelInfo(
        id="qwen/qwen3.8-27b",
        display_name="Qwen 3.8 27B",
        context_window=131_042,
        input_price_per_1m=0.30,
        output_price_per_1m=0.30,
        description="Latest Qwen model",
    ),
    ModelInfo(
        id="allam-2-7b",
        display_name="Allam 2 7B",
        context_window=4_096,
        input_price_per_1m=0.10,
        output_price_per_1m=0.10,
        description="Arabic-focused model",
    ),
    ModelInfo(
        id="groq/compound",
        display_name="Groq Compound",
        context_window=131_072,
        input_price_per_1m=0.50,
        output_price_per_1m=0.50,
        description="Compound model for multi-step reasoning",
    ),
    ModelInfo(
        id="groq/compound-mini",
        display_name="Groq Compound Mini",
        context_window=131_072,
        input_price_per_1m=0.10,
        output_price_per_1m=0.10,
        description="Lightweight compound model",
    ),
)

# ---- Hugging Face Inference Providers (OpenAI-compatible router) ----
# These use the HF router at https://router.huggingface.co/v1 with HF token
HF_MODELS = (
    ModelInfo(
        id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama 3.3 70B (HF Router)",
        context_window=131_072,
        input_price_per_1m=0.59,
        output_price_per_1m=0.79,
        description="Routed via HF to fastest provider (Groq, Together, etc.)",
    ),
    ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B (HF Router)",
        context_window=131_072,
        input_price_per_1m=0.05,
        output_price_per_1m=0.08,
        description="Cheapest Llama via HF router",
    ),
    ModelInfo(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        display_name="Mixtral 8x7B (HF Router)",
        context_window=32_768,
        input_price_per_1m=0.24,
        output_price_per_1m=0.24,
        description="MoE model via HF router",
    ),
    ModelInfo(
        id="google/gemma-2-9b-it",
        display_name="Gemma 2 9B IT (HF Router)",
        context_window=8_192,
        input_price_per_1m=0.20,
        output_price_per_1m=0.20,
        description="Gemma via HF router",
    ),
    ModelInfo(
        id="deepseek-ai/DeepSeek-V3",
        display_name="DeepSeek V3 (HF Router)",
        context_window=128_000,
        input_price_per_1m=0.60,
        output_price_per_1m=1.10,
        description="Top-tier open model via HF router",
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-72B-Instruct",
        display_name="Qwen 2.5 72B (HF Router)",
        context_window=131_072,
        input_price_per_1m=0.60,
        output_price_per_1m=3.00,
        description="Strong multilingual/coding via HF router",
    ),
    ModelInfo(
        id="openai/gpt-oss-120b",
        display_name="GPT-OSS 120B (HF Router)",
        context_window=131_072,
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        description="Open-weight GPT-OSS via HF router",
    ),
)

# ---- OpenAI ----
OPENAI_MODELS = (
    ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        supports_vision=True,
        description="General flagship — complex analysis, code, multimodal",
    ),
    ModelInfo(
        id="gpt-4o-mini",
        display_name="GPT-4o mini",
        context_window=128_000,
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        supports_vision=True,
        description="Best value — 16x cheaper than GPT-4o, 80% capability",
    ),
    ModelInfo(
        id="o1",
        display_name="o1 (Reasoning)",
        context_window=200_000,
        input_price_per_1m=15.00,
        output_price_per_1m=60.00,
        description="Complex reasoning, math, multi-step problems",
    ),
    ModelInfo(
        id="o1-mini",
        display_name="o1-mini (Reasoning)",
        context_window=128_000,
        input_price_per_1m=3.00,
        output_price_per_1m=12.00,
        description="STEM reasoning at lower cost than o1",
    ),
    ModelInfo(
        id="o3-mini",
        display_name="o3-mini (Reasoning)",
        context_window=200_000,
        input_price_per_1m=1.10,
        output_price_per_1m=4.40,
        description="Cost-effective reasoning model",
    ),
    ModelInfo(
        id="gpt-4.1",
        display_name="GPT-4.1",
        context_window=1_000_000,
        input_price_per_1m=2.00,
        output_price_per_1m=8.00,
        supports_vision=True,
        description="Larger context (1M) flagship",
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        context_window=1_000_000,
        input_price_per_1m=0.40,
        output_price_per_1m=1.60,
        supports_vision=True,
        description="Cheaper 1M context model",
    ),
    ModelInfo(
        id="gpt-5.4",
        display_name="GPT-5.4 (Latest Flagship)",
        context_window=1_000_000,
        input_price_per_1m=2.50,
        output_price_per_1m=15.00,
        supports_vision=True,
        description="Newest GPT-5 series flagship",
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        display_name="GPT-5.4 Mini",
        context_window=400_000,
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
        supports_vision=True,
        description="Cost-efficient GPT-5 variant",
    ),
    ModelInfo(
        id="gpt-5.4-nano",
        display_name="GPT-5.4 Nano",
        context_window=128_000,
        input_price_per_1m=0.20,
        output_price_per_1m=1.25,
        description="Ultra-cheap GPT-5 variant",
    ),
)

# ---- Registry ----
ALL_PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="gemini",
        display_name="Google Gemini",
        api_key_env="GEMINI_API_KEY",
        models=GEMINI_MODELS,
        description="Google's multimodal models with 1M context, free tier available",
    ),
    ProviderInfo(
        id="groq",
        display_name="Groq",
        api_key_env="GROQ_API_KEY",
        models=GROQ_MODELS,
        description="Ultra-fast LPU inference for open-weight models (Llama, Mixtral, Gemma)",
    ),
    ProviderInfo(
        id="huggingface",
        display_name="Hugging Face (Inference Providers)",
        api_key_env="HF_TOKEN",
        base_url="https://router.huggingface.co/v1",
        models=HF_MODELS,
        description="OpenAI-compatible router to 15+ providers (Groq, Together, Fireworks, etc.)",
    ),
    ProviderInfo(
        id="openai",
        display_name="OpenAI",
        api_key_env="OPENAI_API_KEY",
        models=OPENAI_MODELS,
        description="Proprietary frontier models (GPT-4o, o1, GPT-5 series)",
    ),
)

PROVIDER_BY_ID = {p.id: p for p in ALL_PROVIDERS}

def get_provider(provider_id: str) -> ProviderInfo | None:
    """Look up a provider by its short ID."""
    return PROVIDER_BY_ID.get(provider_id)

def get_model(provider_id: str, model_id: str) -> ModelInfo | None:
    """Look up a model by provider ID and model ID."""
    provider = get_provider(provider_id)
    if not provider:
        return None
    for m in provider.models:
        if m.id == model_id:
            return m
    return None

def list_models_for_provider(provider_id: str) -> tuple[ModelInfo, ...]:
    """Return all models for a given provider."""
    provider = get_provider(provider_id)
    return provider.models if provider else ()