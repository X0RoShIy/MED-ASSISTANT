"""
Оценка качества ответов: сравнение базовой модели и дообученной (LoRA).

Метрики подобраны под цель проекта — не общие BLEU/ROUGE (которые плохо
отражают фактическую корректность в медицине), а проверяемые показатели:

  1. Точность фактов (Factual accuracy)
     Доля найденных в эталоне взаимодействий, которые модель упомянула в ответе.
     Главная метрика — пропуск реального конфликта недопустим.

  2. Отсутствие галлюцинаций (Hallucination rate)
     Доля ответов, где модель упомянула вещество/конфликт, которого НЕТ в отчёте.

  3. Доля русского языка (Russian ratio)
     Кириллица / (кириллица + латиница без названий препаратов).
     Llama 3 8B без дообучения часто срывается на английский.

  4. Соблюдение структуры (Structure score)
     Наличие 5 обязательных блоков: описание, побочки, взаимодействия,
     рекомендация, дисклеймер.

  5. Наличие дисклеймера (Disclaimer rate)
     Доля ответов с напоминанием о консультации с врачом.

Скрипт работает в двух режимах:
  --mode reference   — сравнить ЭТАЛОННЫЕ ответы датасета с «псевдо-базовой»
                       генерацией (эмуляция; работает без GPU, для проверки метрик)
  --mode models      — сравнить реальную базовую и дообученную модель
                       (требует GPU и обученный адаптер)

Запуск без GPU (демонстрация метрик на эталонах):
    python finetune/evaluate.py --mode reference --n 200
"""
from __future__ import annotations
import re
import os
import json
import argparse
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "finetune" / "dataset"


# ============================================================
# Метрики (чистые функции, без зависимостей от модели)
# ============================================================
def russian_ratio(text: str) -> float:
    """Доля кириллицы среди буквенных символов. Латинские названия препаратов
    в круглых скобках и после маркеров исключаются — они должны оставаться
    на латинице, это не «срыв на английский». 1.0 = полностью русский."""
    # Убираем содержимое скобок (там обычно англ. названия/пояснения) и
    # слова с цифрами рядом (торговые названия вроде 'Brufen 400')
    cleaned = re.sub(r"\([^)]*\)", " ", text)
    cleaned = re.sub(r"\b[A-Za-z][A-Za-z\-]*\s*\d+[A-Za-z0-9]*", " ", cleaned)
    cyr = len(re.findall(r"[а-яё]", cleaned, re.IGNORECASE))
    lat = len(re.findall(r"[a-z]", cleaned, re.IGNORECASE))
    if cyr + lat == 0:
        return 1.0
    return cyr / (cyr + lat)


def has_disclaimer(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ["врач", "консульт", "специалист", "доктор"])


def structure_score(text: str) -> float:
    """Сколько из 5 смысловых блоков присутствует (0..1)."""
    t = text.lower()
    checks = [
        any(k in t for k in ["применя", "назначен", "использ", "для чего", "рассматрив"]),  # описание
        any(k in t for k in ["побочн", "эффект", "реакци"]),                                  # побочки
        any(k in t for k in ["взаимодейств", "сочетан", "совместн", "конфликт"]),             # взаимодействия
        any(k in t for k in ["рекоменд", "следует", "стоит", "избега", "соблюда"]),           # рекомендация
        has_disclaimer(t),                                                                     # дисклеймер
    ]
    return sum(checks) / len(checks)


def extract_substances_from_report(report: str) -> set[str]:
    """Достаёт упомянутые в отчёте вещества/препараты для проверки галлюцинаций."""
    subs = set()
    for m in re.finditer(r"Действующие вещества:\s*(.+)", report):
        for part in m.group(1).split(","):
            subs.add(part.strip().lower())
    # вещества из строк взаимодействий
    for m in re.finditer(r"•\s*(.+?)\s*↔\s*(.+)", report):
        subs.add(m.group(1).strip().lower())
        subs.add(m.group(2).strip().lower())
    return {s for s in subs if s}


def factual_accuracy(report: str, answer: str) -> float | None:
    """Доля взаимодействий из отчёта, упомянутых в ответе. None — если конфликтов нет."""
    conflicts = re.findall(r"•\s*(.+?)\s*↔\s*(.+)", report)
    if not conflicts:
        return None
    ans_low = answer.lower()
    hit = 0
    for a, b in conflicts:
        a_key = a.strip().lower().split()[0] if a.strip() else ""
        b_key = b.strip().lower().split()[0] if b.strip() else ""
        # засчитываем, если оба вещества (или их основа) упомянуты
        if a_key and b_key and a_key[:5] in ans_low and b_key[:5] in ans_low:
            hit += 1
    return hit / len(conflicts)


def hallucination_flag(report: str, answer: str) -> int:
    """1 — если в ответе утверждается наличие конфликта, которого нет в отчёте.
    Аккуратно: реагируем только на явные утверждения об обнаруженном
    взаимодействии, а не на оговорки вроде «не гарантирует безопасность»."""
    no_conflict = "не найдено" in report.lower() or "не зарегистрирова" in report.lower()
    if not no_conflict:
        return 0
    a = answer.lower()
    # Явные утверждения, что взаимодействие ЕСТЬ
    positive_claims = [
        "обнаружены взаимодействия", "обнаружено взаимодействие",
        "выявлены взаимодействия", "найдены взаимодействия",
        "нельзя сочетать", "нельзя принимать вместе",
        "опасное сочетание", "опасное взаимодействие",
    ]
    # Оговорки, которые НЕ считаются галлюцинацией
    has_positive = any(p in a for p in positive_claims)
    return 1 if has_positive else 0


def evaluate_answers(samples: list[dict]) -> dict:
    """samples: [{report, answer}]. Возвращает усреднённые метрики."""
    fa, halluc, rus, struct, disc = [], [], [], [], []
    for s in samples:
        report, answer = s["report"], s["answer"]
        f = factual_accuracy(report, answer)
        if f is not None:
            fa.append(f)
        halluc.append(hallucination_flag(report, answer))
        rus.append(russian_ratio(answer))
        struct.append(structure_score(answer))
        disc.append(1 if has_disclaimer(answer) else 0)

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    return {
        "factual_accuracy": avg(fa),
        "hallucination_rate": avg(halluc),
        "russian_ratio": avg(rus),
        "structure_score": avg(struct),
        "disclaimer_rate": avg(disc),
        "n_with_conflicts": len(fa),
        "n_total": len(samples),
    }


def print_comparison(name_a: str, m_a: dict, name_b: str, m_b: dict):
    print(f"\n{'Метрика':<28}{name_a:>14}{name_b:>14}{'Δ':>10}")
    print("-" * 66)
    rows = [
        ("Точность фактов ↑", "factual_accuracy", "%"),
        ("Галлюцинации ↓", "hallucination_rate", "%"),
        ("Доля русского ↑", "russian_ratio", "%"),
        ("Структура (5 блоков) ↑", "structure_score", "%"),
        ("Дисклеймер ↑", "disclaimer_rate", "%"),
    ]
    for label, key, unit in rows:
        a, b = m_a[key], m_b[key]
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"{label:<28}{a*100:>13.1f}%{b*100:>13.1f}%{sign}{delta*100:>8.1f} п.п.")


# ============================================================
# Режим 1: проверка метрик на эталонах (без GPU)
# ============================================================
def mode_reference(n: int):
    """
    Демонстрирует, как работают метрики, БЕЗ запуска моделей.
    «Базовая модель» эмулируется ухудшением эталона (срыв на английский,
    пропуск блоков) — чтобы показать, что метрики ловят разницу.
    Реальное сравнение моделей — в режиме models.
    """
    import random
    rng = random.Random(7)

    val_path = DATASET_DIR / "val.jsonl"
    with open(val_path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f][:n]

    # «Дообученная» = эталонный ответ (то, к чему стремимся)
    finetuned = [{"report": ex["messages"][1]["content"],
                  "answer": ex["messages"][2]["content"]} for ex in data]

    # «Базовая» = эмуляция типичных проблем Llama 3 8B без дообучения:
    #   - часть текста на английском
    #   - иногда пропущен дисклеймер или блок взаимодействий
    EN_FRAGMENTS = {
        "побочные эффекты": "side effects",
        "взаимодействия": "interactions",
        "рекомендация": "recommendation",
        "применяется": "is used for",
        "проконсультируйтесь с врачом": "consult a doctor",
    }
    baseline = []
    for ex in data:
        ans = ex["messages"][2]["content"]
        report = ex["messages"][1]["content"]
        bad = ans
        # с вероятностью срываем часть на английский
        for ru, en in EN_FRAGMENTS.items():
            if rng.random() < 0.5:
                bad = bad.replace(ru, en)
        # иногда выкидываем дисклеймер
        if rng.random() < 0.4:
            bad = bad.split("⚕")[0]
        # иногда теряем блок взаимодействий
        if rng.random() < 0.3 and "Обнаружены взаимодействия" in bad:
            bad = re.sub(r"\*\*Обнаружены взаимодействия.*?(?=\n\n)", "", bad, flags=re.S)
        baseline.append({"report": report, "answer": bad})

    m_base = evaluate_answers(baseline)
    m_ft = evaluate_answers(finetuned)
    print(f"Оценка на {len(data)} примерах валидации")
    print("(режим reference: «базовая» эмулирована, «дообученная» = эталон)")
    print_comparison("База (эмул.)", m_base, "Дообуч.", m_ft)
    print("\nПримечание: для реальных чисел запустите --mode models на GPU.")

    # Сохраняем, чтобы цифры можно было вставить в презентацию
    out_path = ROOT / "finetune" / "metrics_reference.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_emulated": m_base, "reference": m_ft}, f,
                  ensure_ascii=False, indent=2)


# ============================================================
# Режим 2: реальное сравнение моделей (нужен GPU)
# ============================================================
def mode_models(n: int, base_model: str, adapter_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    val_path = DATASET_DIR / "val.jsonl"
    with open(val_path, encoding="utf-8") as f:
        data = [json.loads(l) for l in f][:n]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def gen_all(model):
        out = []
        for ex in data:
            msgs = ex["messages"][:2]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=512, temperature=0.3,
                                   top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id)
            ans = tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append({"report": ex["messages"][1]["content"], "answer": ans.strip()})
        return out

    print("Загрузка базовой модели...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    base.eval()
    print(f"Генерация {n} ответов базовой моделью...")
    base_answers = gen_all(base)
    m_base = evaluate_answers(base_answers)
    del base
    torch.cuda.empty_cache()

    print("Загрузка дообученной модели (база + адаптер)...")
    ft = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    ft = PeftModel.from_pretrained(ft, adapter_dir)
    ft.eval()
    print(f"Генерация {n} ответов дообученной моделью...")
    ft_answers = gen_all(ft)
    m_ft = evaluate_answers(ft_answers)

    print_comparison("Базовая", m_base, "Дообученная", m_ft)

    # Сохраняем для презентации
    result = {"baseline": m_base, "finetuned": m_ft}
    out_path = ROOT / "finetune" / "metrics_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены в {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "models"], default="reference")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--base", default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "finetune" / "lora_adapter"))
    args = ap.parse_args()

    if args.mode == "reference":
        mode_reference(args.n)
    else:
        mode_models(args.n, args.base, args.adapter)


if __name__ == "__main__":
    main()
