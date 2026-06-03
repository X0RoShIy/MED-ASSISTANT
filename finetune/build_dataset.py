"""
Генерация обучающего датасета для дообучения (QLoRA) языковой модели.

Идея: из баз данных (Medicine_Details + db_drug_interactions) автоматически
строятся пары «инструкция → эталонный ответ». Модель учится превращать
структурированный технический отчёт в связную русскоязычную рекомендацию
с правильной структурой, тоном и дисклеймером — то, что трудно стабильно
получить одним лишь промптом на Llama 3 8B.

Формат выхода — JSONL в chat-формате (messages: system/user/assistant),
совместимый с TRL SFTTrainer и большинством фреймворков дообучения.

Запуск:
    python finetune/build_dataset.py            # генерирует ~2000 примеров
    python finetune/build_dataset.py --n 5000   # больше примеров
"""
from __future__ import annotations
import sys
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_preparation import load_cache
from medical_glossary import translate_medical_terms

SYSTEM_PROMPT = (
    "Ты — русскоязычный медицинский ассистент. На основе ТОЛЬКО предоставленных "
    "данных из справочников объясняешь простым языком назначение препарата, "
    "побочные эффекты и взаимодействия. Не выдумываешь факты. Не назначаешь "
    "дозировки. Всегда напоминаешь о необходимости консультации с врачом."
)

# Шаблоны вопросов пользователя — для разнообразия инструкций
USER_QUESTIONS_PAIR = [
    "Можно ли принимать эти препараты вместе?",
    "Безопасно ли это сочетание?",
    "Какие риски при совместном приёме?",
    "Не опасно ли пить их одновременно?",
    "Стоит ли беспокоиться о взаимодействии?",
]
USER_QUESTIONS_SINGLE = [
    "Расскажи об этом препарате.",
    "Для чего применяется это лекарство и какие у него побочные эффекты?",
    "Что нужно знать перед приёмом?",
    "Какие меры предосторожности при приёме?",
]


def _clean_field(text: str) -> str:
    """Чинит слипшиеся значения и переводит термины.
    Побочки в исходных данных идут как 'Vomiting Nausea Diarrhea' (каждое с заглавной,
    разделены пробелом) — превращаем в список через запятую, затем переводим."""
    if not text:
        return ""
    import re
    # camelCase-склейка: 'reliefTreatment' -> 'relief, Treatment'
    text = re.sub(r"([a-zа-я])([A-ZА-Я])", r"\1, \2", text)
    # Перевод медицинских терминов
    text = translate_medical_terms(text)
    # Разбиваем на токены и склеиваем через запятую там, где идут
    # несколько слов с заглавной буквы подряд (список симптомов)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_report(drugs_info: list[dict], conflicts: list[dict],
                   question: str) -> str:
    """Строит технический отчёт в том же формате, что и рабочий pipeline."""
    lines = [f"Анализируемые препараты ({len(drugs_info)}):"]
    for i, d in enumerate(drugs_info, 1):
        lines.append(f"\n[{i}] {d['name']}")
        if d.get("substances"):
            lines.append(f"    Действующие вещества: {', '.join(d['substances'])}")
        if d.get("uses"):
            lines.append(f"    Применение: {_clean_field(d['uses'])}")
        if d.get("side_effects"):
            lines.append(f"    Побочные эффекты: {_clean_field(d['side_effects'])}")
    if conflicts:
        lines.append(f"\nНайденные взаимодействия ({len(conflicts)}):")
        for c in conflicts:
            lines.append(f"  • {c['drug_a']} ↔ {c['drug_b']}")
            lines.append(f"    {_clean_field(c['description'])}")
    else:
        lines.append("\nВзаимодействий между указанными препаратами не найдено.")
    lines.append(f"\nЗапрос пользователя: {question}")
    return "\n".join(lines)


def _build_reference_answer(drugs_info: list[dict], conflicts: list[dict],
                           single: bool) -> str:
    """
    Строит ЭТАЛОННЫЙ ответ из данных детерминированно (без LLM).
    Это «золотой стандарт» для обучения: структура из 5 блоков,
    русский язык, дисклеймер. Модель учится воспроизводить именно
    такой формат и тон.
    """
    parts = []

    # 1. Описание
    if single:
        d = drugs_info[0]
        uses = _clean_field(d.get("uses", "")) or "сведения о применении отсутствуют в справочнике"
        parts.append(f"**{d['name']}** применяется в следующих случаях: {uses.lower()}.")
    else:
        names = ", ".join(d["name"] for d in drugs_info)
        parts.append(f"Рассматривается совместный приём препаратов: {names}.")

    # 2. Побочные эффекты
    se_all = []
    for d in drugs_info:
        se = _clean_field(d.get("side_effects", ""))
        if se:
            se_all.append(f"{d['name']} — {se.lower()}")
    if se_all:
        parts.append("**Возможные побочные эффекты:** " + "; ".join(se_all) + ".")

    # 3. Взаимодействия
    if conflicts:
        conf_lines = ["**Обнаружены взаимодействия — отнеситесь внимательно:**"]
        for c in conflicts:
            desc = _clean_field(c["description"])
            conf_lines.append(f"• {c['drug_a']} и {c['drug_b']}: {desc.lower()}")
        parts.append("\n".join(conf_lines))
    elif not single:
        parts.append("**Взаимодействия:** в справочнике опасных сочетаний между "
                     "указанными препаратами не зарегистрировано. Это не гарантирует "
                     "полную безопасность — данные ограничены справочником.")

    # 4. Рекомендации
    if conflicts:
        parts.append("**Рекомендация:** перед совместным приёмом обязательно "
                     "проконсультируйтесь с врачом — возможно, потребуется "
                     "коррекция схемы лечения или замена препарата.")
    else:
        parts.append("**Рекомендация:** соблюдайте инструкцию по применению и "
                     "следите за самочувствием.")

    # 5. Дисклеймер
    parts.append("⚕ Эта информация носит справочный характер. Окончательное решение "
                 "о приёме лекарств принимает только лечащий врач.")

    return "\n\n".join(parts)


def generate_examples(n: int, cache_dir: str, seed: int = 42) -> list[dict]:
    medicine_index, interaction_index, alias_map = load_cache(cache_dir)
    rng = random.Random(seed)

    # Препараты, у которых есть канонические вещества (для поиска взаимодействий)
    usable = [rec for rec in medicine_index.values() if rec["substances_canonical"]]

    # Пул реальных пар взаимодействий — берём уникальные
    interaction_pairs = list(interaction_index.items())

    examples = []
    attempts = 0
    max_attempts = n * 20

    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        # ~60% примеров — пары с взаимодействием, 25% — одиночные, 15% — пары без конфликта
        roll = rng.random()

        if roll < 0.60:
            # Пара с РЕАЛЬНЫМ взаимодействием из базы
            (sub_a, sub_b), desc = rng.choice(interaction_pairs)
            # Находим препараты, содержащие эти вещества
            rec_a = next((r for r in usable if sub_a in [s.lower() for s in r["substances_canonical"]]), None)
            rec_b = next((r for r in usable if sub_b in [s.lower() for s in r["substances_canonical"]]), None)
            if not rec_a or not rec_b or rec_a["original_name"] == rec_b["original_name"]:
                continue
            drugs_info = [
                {"name": rec_a["original_name"], "substances": rec_a["substances_raw"],
                 "uses": rec_a["uses"], "side_effects": rec_a["side_effects"]},
                {"name": rec_b["original_name"], "substances": rec_b["substances_raw"],
                 "uses": rec_b["uses"], "side_effects": rec_b["side_effects"]},
            ]
            conflicts = [{"drug_a": sub_a.title(), "drug_b": sub_b.title(), "description": desc}]
            question = rng.choice(USER_QUESTIONS_PAIR)
            single = False

        elif roll < 0.85:
            # Одиночный препарат
            rec = rng.choice(usable)
            drugs_info = [{"name": rec["original_name"], "substances": rec["substances_raw"],
                          "uses": rec["uses"], "side_effects": rec["side_effects"]}]
            conflicts = []
            question = rng.choice(USER_QUESTIONS_SINGLE)
            single = True

        else:
            # Пара БЕЗ известного взаимодействия
            rec_a, rec_b = rng.sample(usable, 2)
            subs_a = [s.lower() for s in rec_a["substances_canonical"]]
            subs_b = [s.lower() for s in rec_b["substances_canonical"]]
            has_conflict = any((a, b) in interaction_index or (b, a) in interaction_index
                              for a in subs_a for b in subs_b)
            if has_conflict:
                continue  # этот случай попадёт в ветку с конфликтом
            drugs_info = [
                {"name": rec_a["original_name"], "substances": rec_a["substances_raw"],
                 "uses": rec_a["uses"], "side_effects": rec_a["side_effects"]},
                {"name": rec_b["original_name"], "substances": rec_b["substances_raw"],
                 "uses": rec_b["uses"], "side_effects": rec_b["side_effects"]},
            ]
            conflicts = []
            question = rng.choice(USER_QUESTIONS_PAIR)
            single = False

        report = _format_report(drugs_info, conflicts, question)
        answer = _build_reference_answer(drugs_info, conflicts, single)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": report},
                {"role": "assistant", "content": answer},
            ]
        })

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="число примеров")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "finetune" / "dataset"))
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    print(f"Генерация {args.n} обучающих примеров...")
    examples = generate_examples(args.n, args.cache)
    print(f"Сгенерировано: {len(examples)}")

    rng = random.Random(123)
    rng.shuffle(examples)
    n_val = int(len(examples) * args.val_split)
    val, train = examples[:n_val], examples[n_val:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} → {path}")

    # Пример для наглядности
    print("\n=== Пример обучающей пары ===")
    ex = train[0]
    print("[USER]:", ex["messages"][1]["content"][:300])
    print("\n[ASSISTANT]:", ex["messages"][2]["content"][:400])


if __name__ == "__main__":
    main()
