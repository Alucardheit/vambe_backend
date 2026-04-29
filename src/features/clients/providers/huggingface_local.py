"""
Hugging Face local provider — corre el modelo en la máquina del usuario.

Soporta:
  - macOS Apple Silicon → MPS (Metal Performance Shaders)
  - Windows / Linux con NVIDIA → CUDA
  - Cualquier otro caso → CPU

Requiere instalar las deps opcionales:
    uv sync --extra huggingface

El modelo se descarga al cache local de HF (`~/.cache/huggingface` o `HF_HOME`)
la primera vez. Tamaños típicos: SmolLM2-1.7B-Instruct → ~3.4 GB FP16.
"""

import asyncio
import json
import logging
import os
import re
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

# Acelera descargas de Hugging Face Hub (Rust, conexiones paralelas).
# Se setea antes de cualquier import de transformers/huggingface_hub.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from .base import LLMProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _detect_device() -> str:
    """
    Selección explícita por plataforma:
      - macOS  → MPS si está disponible (Apple Silicon), si no CPU.
      - Otros  → CUDA si hay NVIDIA, si no CPU.
    CUDA nunca está disponible en macOS, por eso no se chequea ahí.
    """
    import torch

    if sys.platform == "darwin":
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            logger.info("Plataforma macOS detectada — usando MPS.")
            return "mps"
        logger.info("macOS sin MPS disponible (probablemente Intel Mac) — usando CPU.")
        return "cpu"

    if torch.cuda.is_available():
        logger.info("CUDA detectado — usando GPU NVIDIA.")
        return "cuda"

    logger.info("Sin GPU disponible — usando CPU.")
    return "cpu"


def _select_dtype(device: str) -> Any:
    """fp16 en GPU, fp32 en CPU (CPU no soporta fp16 eficientemente)."""
    import torch

    if device == "cpu":
        return torch.float32
    if device == "mps":
        # MPS no soporta bien bf16; usar fp16.
        return torch.float16
    return torch.float16


@lru_cache(maxsize=2)
def _load_hf_model(model_id: str, token: str | None) -> tuple[Any, Any, str]:
    """Carga (tokenizer, model, device) y los cachea — singleton por (model_id, token)."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Faltan deps de Hugging Face. Instala con: `uv sync --extra huggingface`"
        ) from exc

    device = _detect_device()
    dtype = _select_dtype(device)

    logger.info("Cargando modelo HF '%s' en %s (dtype=%s)…", model_id, device, dtype)
    auth: dict[str, Any] = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_id, **auth)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **auth,
    ).to(device)
    model.eval()
    logger.info("Modelo HF '%s' listo.", model_id)
    return tokenizer, model, device


def _extract_first_json(text: str) -> str | None:
    """Extrae el primer objeto JSON balanceado `{...}` del texto."""
    # Quita fences markdown comunes: ```json ... ``` o ``` ... ```
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if start == -1:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


class HuggingFaceLocalProvider(LLMProvider):
    name = "huggingface"
    # GPU/CPU es single-threaded para inferencia: serializar requests evita OOM.
    recommended_concurrency = 1

    def __init__(self, model_id: str, token: str | None = None) -> None:
        self.model_id = model_id
        self.token = token or None
        # Lock async para serializar generaciones (tokenizer/model no son thread-safe en GPU).
        self._lock = asyncio.Lock()

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, system, user, schema, temperature)

    def _generate_sync(self, system: str, user: str, schema: type[T], temperature: float) -> T:
        import torch

        tokenizer, model, device = _load_hf_model(self.model_id, self.token)

        # Schema compacto: solo nombres + tipos, sin descripciones (ahorra ~40% tokens).
        compact_fields = {
            name: field.annotation.__name__ if hasattr(field.annotation, "__name__") else "str"
            for name, field in schema.model_fields.items()
        }
        user_with_schema = (
            f"{user}\n\n"
            f"Respond with a single JSON object containing exactly these string fields: "
            f"{json.dumps(compact_fields, ensure_ascii=False)}. "
            f"No markdown fences, no commentary, no text outside the JSON."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_with_schema},
        ]

        # Llama-style chat template (SmolLM2-Instruct lo trae).
        # transformers v5: apply_chat_template devuelve BatchEncoding cuando
        # return_dict=True; pasamos **inputs a generate() para incluir
        # attention_mask y evitar el AttributeError sobre .shape.
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
            inputs = tokenizer(text, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Greedy decoding (do_sample=False) es ~2x más rápido que sampling y
        # más determinístico para JSON. max_new_tokens=768 cubre los ~500-700
        # tokens que ocupan los 12 campos del schema.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, input_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        json_str = _extract_first_json(text)
        if not json_str:
            raise ValueError(f"El modelo HF no produjo JSON válido. Salida: {text[:300]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON malformado del modelo HF: {exc}. Texto: {json_str[:300]}"
            ) from exc

        return schema.model_validate(data)
