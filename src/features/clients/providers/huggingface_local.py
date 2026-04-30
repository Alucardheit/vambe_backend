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

# Tokens reservados para la respuesta JSON. El resto del contexto es para el input.
_MAX_NEW_TOKENS = 3072
# Contexto máximo del modelo. SmolLM2-1.7B soporta hasta 8192.
_MODEL_MAX_CONTEXT = 8192
# Tokens disponibles para el input (contexto - output reservado - margen de seguridad).
_MAX_INPUT_TOKENS = _MODEL_MAX_CONTEXT - _MAX_NEW_TOKENS - 64


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
        dtype=dtype,
        device_map=device,
        **auth,
    )
    model.eval()
    logger.info("Modelo HF '%s' listo en %s.", model_id, device)
    return tokenizer, model, device


def _extract_first_json(text: str) -> str | None:
    """
    Extrae el primer objeto JSON balanceado `{...}` del texto.

    Maneja dos casos:
    - Fence completo: ```json ... ```
    - Fence sin cierre (output truncado): ```json ... <EOF>
    """
    # Intenta fence completo primero.
    fence_closed = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_closed:
        text = fence_closed.group(1)
    else:
        # Si hay apertura de fence sin cierre, descarta solo el marcador inicial.
        fence_open = re.match(r"```(?:json)?\s*\n?", text, re.IGNORECASE)
        if fence_open:
            text = text[fence_open.end():]

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
    # El lock interno ya serializa la GPU — no necesitamos semáforo adicional.
    recommended_concurrency = 1

    def __init__(self, model_id: str, token: str | None = None) -> None:
        self.model_id = model_id
        self.token = token or None
        # Lock para serializar generaciones: tokenizer/model no son thread-safe.
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

        # Lista de nombres de campos, sin tipos (evita que el modelo eche "str" como valor).
        field_names = list(schema.model_fields.keys())
        user_with_schema = (
            f"{user}\n\n"
            f"Respond with a single JSON object with EXACTLY these keys (in this order): "
            f"{', '.join(field_names)}. "
            f"Each value must be a string with the actual analysis CONTENT — never the field "
            f"name, never 'str', never the example words from the instructions. "
            f"Rules: output ONLY the raw JSON object, no markdown fences, no extra text. "
            f"Keep each field value under 25 words. Start your response with {{."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_with_schema},
        ]

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

        input_length = inputs["input_ids"].shape[1]

        # Truncar si el input supera el límite: recorta desde el centro del mensaje
        # de usuario (donde está la transcripción), preservando sistema y esquema.
        if input_length > _MAX_INPUT_TOKENS:
            logger.warning(
                "Input demasiado largo (%d tokens), truncando a %d tokens.",
                input_length,
                _MAX_INPUT_TOKENS,
            )
            for key in inputs:
                inputs[key] = inputs[key][:, :_MAX_INPUT_TOKENS]
            input_length = _MAX_INPUT_TOKENS

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, input_length:]
        n_new = len(new_tokens)
        if n_new >= _MAX_NEW_TOKENS - 10:
            logger.warning("Output cerca del límite (%d/%d tokens). JSON puede estar truncado.", n_new, _MAX_NEW_TOKENS)
        else:
            logger.debug("Generados %d tokens nuevos.", n_new)
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
