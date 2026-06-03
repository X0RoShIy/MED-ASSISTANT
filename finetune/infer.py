"""
Инференс дообученной модели (база + LoRA-адаптер).

Два способа использования:

1) Прямой инференс через transformers (для теста адаптера):
       python finetune/infer.py --report "текст технического отчёта"

2) Экспорт в Ollama (для интеграции в основное приложение):
   Адаптер сливается с базой и конвертируется в GGUF — см. функцию export_for_ollama
   и инструкцию в finetune/README.md.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "finetune" / "lora_adapter"
DEFAULT_BASE = "NousResearch/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = (
    "Ты — русскоязычный медицинский ассистент. На основе ТОЛЬКО предоставленных "
    "данных из справочников объясняешь простым языком назначение препарата, "
    "побочные эффекты и взаимодействия. Не выдумываешь факты. Не назначаешь "
    "дозировки. Всегда напоминаешь о необходимости консультации с врачом."
)


def load_model(base_model: str, adapter_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir)  # подключаем адаптер
    model.eval()
    return model, tok


def generate(model, tok, report_text: str, max_new_tokens: int = 512) -> str:
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report_text},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, top_p=0.9, do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def export_for_ollama(base_model: str, adapter_dir: str, out_dir: str):
    """
    Сливает LoRA-адаптер с базовой моделью и сохраняет в полном виде.
    Далее (вне Python) конвертируется в GGUF через llama.cpp:

        python llama.cpp/convert_hf_to_gguf.py <out_dir> --outfile med-llama.gguf
        ollama create med-assistant -f Modelfile

    где Modelfile содержит:
        FROM ./med-llama.gguf
        SYSTEM "..."
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Загрузка базовой модели в fp16 (без квантизации для слияния)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("Слияние адаптера с базой...")
    model = model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    print(f"✓ Слитая модель сохранена в {out_dir}")
    print("Далее конвертируйте в GGUF (см. инструкцию в docstring и README).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=str(ADAPTER_DIR))
    ap.add_argument("--report", help="текст технического отчёта для теста")
    ap.add_argument("--export-ollama", metavar="OUT_DIR",
                    help="слить адаптер с базой и сохранить для Ollama")
    args = ap.parse_args()

    if args.export_ollama:
        export_for_ollama(args.base, args.adapter, args.export_ollama)
        return

    model, tok = load_model(args.base, args.adapter)
    report = args.report or (
        "Анализируемые препараты (2):\n\n"
        "[1] Aspirin\n    Действующие вещества: Acetylsalicylic acid\n"
        "    Применение: профилактика инфаркта\n"
        "[2] Ibuprofen\n    Действующие вещества: Ibuprofen\n\n"
        "Найденные взаимодействия (1):\n"
        "  • Acetylsalicylic acid ↔ Ibuprofen\n"
        "    риск или тяжесть побочных эффектов могут возрастать\n\n"
        "Запрос пользователя: Можно ли принимать вместе?"
    )
    print("=== ОТЧЁТ ===\n", report)
    print("\n=== ОТВЕТ ДООБУЧЕННОЙ МОДЕЛИ ===\n", generate(model, tok, report))


if __name__ == "__main__":
    main()
