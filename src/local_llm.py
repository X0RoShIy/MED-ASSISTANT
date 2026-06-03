"""
Локальный бэкенд для дообученной модели (Qwen2.5-3B + LoRA-адаптер).

В отличие от llm_client.py (который ходит в Ollama по HTTP), этот модуль
грузит модель прямо в память через transformers + peft и использует
обученный LoRA-адаптер с диска.

Зачем отдельный модуль: приложение по умолчанию работает через Ollama,
но дообученный адаптер — в формате HuggingFace/PEFT, и проще подключить
его напрямую, чем конвертировать в GGUF для Ollama.

Использование в приложении:
    from local_llm import generate_advice_local, is_available
    if is_available():
        resp = generate_advice_local(report_text, user_prompt_text=...)

Модель грузится ОДИН РАЗ при первом вызове и кэшируется (ленивая загрузка),
чтобы не перезагружать её на каждый запрос.

ВАЖНО: требует установленных torch, transformers, peft и наличия адаптера
на диске. Путь к адаптеру задаётся через ADAPTER_DIR или переменную окружения
MED_ADAPTER_DIR.
"""
from __future__ import annotations
import os
from pathlib import Path

# Базовая модель и адаптер. Адаптер — это папка, скачанная с Google Drive
# (med_assistant_lora) и положенная рядом с проектом.
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = os.environ.get(
    "MED_ADAPTER_DIR",
    str(Path(__file__).resolve().parent.parent / "finetune" / "med_assistant_lora"),
)

SYSTEM_PROMPT_RU = (
    "Ты — русскоязычный медицинский ассистент. На основе ТОЛЬКО предоставленных "
    "данных из справочников объясняешь простым языком назначение препарата, "
    "побочные эффекты и взаимодействия. Не выдумываешь факты. Не назначаешь "
    "дозировки. Всегда напоминаешь о необходимости консультации с врачом."
)

# Ленивый кэш загруженной модели — грузим один раз
_model = None
_tokenizer = None
_load_error = None


def is_available() -> bool:
    """Проверяет, можно ли использовать локальный бэкенд:
    установлены библиотеки И существует папка адаптера."""
    if not Path(ADAPTER_DIR).is_dir():
        return False
    try:
        import torch  # noqa
        import transformers  # noqa
        import peft  # noqa
        return True
    except ImportError:
        return False


def _ensure_loaded():
    """Загружает модель и адаптер при первом обращении (и кэширует)."""
    global _model, _tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"[local_llm] Загрузка {BASE_MODEL} + адаптер из {ADAPTER_DIR} ...",
              flush=True)

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # fp16 на GPU; на CPU — float32 (медленно, но работает)
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        device_map = {"": 0} if use_cuda else None

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, device_map=device_map,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model.eval()
        if not use_cuda:
            model = model.to("cpu")

        _model = model
        _tokenizer = tok
        print("[local_llm] Модель загружена.", flush=True)
    except Exception as e:
        _load_error = str(e)
        print(f"[local_llm] Ошибка загрузки: {e}", flush=True)


def generate_advice_local(report_text: str,
                          user_prompt_text: str = "",
                          temperature: float = 0.3,
                          max_new_tokens: int = 512):
    """
    Генерирует совет дообученной моделью. Возвращает объект с полями
    .text, .used_fallback, .error — совместимый с LLMResponse из llm_client.
    """
    # Переиспользуем структуру ответа из основного клиента
    try:
        from llm_client import LLMResponse
    except ImportError:
        from dataclasses import dataclass, field
        @dataclass
        class LLMResponse:
            text: str = ""
            used_fallback: bool = False
            error: str = ""

    _ensure_loaded()
    if _load_error is not None:
        return LLMResponse(text="", used_fallback=True,
                           error=f"Локальная модель недоступна: {_load_error}")

    # Предперевод медтерминов (как в основном клиенте) — стабилизирует русский
    try:
        from medical_glossary import translate_medical_terms
        report_text = translate_medical_terms(report_text)
    except ImportError:
        pass

    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RU},
        {"role": "user", "content": report_text},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.9, do_sample=True,
            pad_token_id=_tokenizer.pad_token_id,
        )
    text = _tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    text = text.strip()

    # Постобработка: добиваем оставшиеся английские медтермины через глоссарий.
    # Модель могла усвоить непереведённые термины из обучающих данных —
    # этот проход чистит остатки латиницы в русском ответе.
    try:
        from medical_glossary import translate_medical_terms
        import re as _re
        latin = len(_re.findall(r"[A-Za-z]", text))
        cyr = len(_re.findall(r"[\u0400-\u04FF]", text))
        if cyr > 0 and latin / max(cyr, 1) > 0.05:
            text = translate_medical_terms(text)
    except ImportError:
        pass

    return LLMResponse(text=text, used_fallback=False)


if __name__ == "__main__":
    # Быстрый тест
    print("Адаптер найден:", Path(ADAPTER_DIR).is_dir(), "->", ADAPTER_DIR)
    print("Бэкенд доступен:", is_available())
    if is_available():
        report = (
            "Анализируемые препараты (2):\n\n"
            "[1] Aspirin\n    Действующие вещества: Acetylsalicylic acid\n"
            "[2] Ibuprofen\n    Действующие вещества: Ibuprofen\n\n"
            "Найденные взаимодействия (1):\n"
            "  • Acetylsalicylic acid \u2194 Ibuprofen\n"
            "    риск побочных эффектов возрастает\n\n"
            "Запрос пользователя: Можно ли принимать вместе?"
        )
        resp = generate_advice_local(report, "Можно ли принимать вместе?")
        print("\n=== ОТВЕТ ДООБУЧЕННОЙ МОДЕЛИ ===\n", resp.text)
