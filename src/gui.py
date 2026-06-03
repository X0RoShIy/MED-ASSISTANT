"""
Графический интерфейс на Gradio.
Запуск:
    python src/gui.py
Откроется http://localhost:7860
"""
from __future__ import annotations
import sys
import random
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender import MedAssistant
from llm_client import list_models, DEFAULT_MODEL, DEFAULT_HOST


BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / 'cache'

_assistant: MedAssistant | None = None


# ============================================================
# Готовые демо-сценарии для кнопки «Пример работы»
# ============================================================
DEMO_SCENARIOS = [
    {
        'title': 'Аспирин + Ибупрофен',
        'drug_list': 'Аспирин\nИбупрофен',
        'prompt': 'Можно ли принимать вместе для облегчения боли и жара?',
    },
    {
        'title': 'Аугментин + Пробенецид',
        'drug_list': 'Augmentin 625\nProbenecid',
        'prompt': 'Безопасно ли принимать эти препараты одновременно?',
    },
    {
        'title': 'Варфарин + Аспирин + Ибупрофен',
        'drug_list': 'Warfarin\nAspirin\nIbuprofen',
        'prompt': 'Какие риски при совместном приёме этих препаратов? '
                  'Я принимаю варфарин по назначению врача.',
    },
    {
        'title': 'Метформин + Аторвастатин',
        'drug_list': 'Метформин\nАторвастатин',
        'prompt': 'Можно ли совмещать эти препараты при сахарном диабете и высоком холестерине?',
    },
    {
        'title': 'Один препарат: что нельзя с Парацетамолом',
        'drug_list': 'Парацетамол',
        'prompt': 'С какими лекарствами Парацетамол лучше не сочетать?',
    },
]


def get_assistant() -> MedAssistant:
    global _assistant
    if _assistant is None:
        _assistant = MedAssistant(cache_dir=CACHE_DIR)
    return _assistant


def analyze_handler(image, user_prompt, drug_list_text, model_name, languages):
    """Обработчик кнопки «Анализировать»."""
    asst = get_assistant()

    image_path = image if isinstance(image, str) else None
    lang_tuple = tuple(languages) if languages else ('en', 'ru')

    # Парсим список препаратов: по строкам, потом по запятым/точкам с запятой
    drug_queries = []
    if drug_list_text:
        for line in drug_list_text.split('\n'):
            for part in line.replace(';', ',').split(','):
                part = part.strip()
                if part:
                    drug_queries.append(part)

    if not image_path and not drug_queries:
        empty_msg = (
            "ℹ️ Укажите хотя бы один препарат: загрузите фото упаковки или "
            "введите названия в текстовое поле слева."
        )
        return "(не загружено)", empty_msg, "_Препараты не указаны_", empty_msg

    # Метка дообученной модели → внутренний идентификатор "finetuned"
    selected_model = model_name or DEFAULT_MODEL
    if selected_model == FINETUNED_LABEL:
        selected_model = "finetuned"

    try:
        result = asst.analyze_multi(
            drug_queries=drug_queries,
            user_prompt=user_prompt or '',
            image_path=image_path,
            ocr_languages=lang_tuple,
            model=selected_model,
        )
    except Exception as e:
        import traceback
        return (
            f"❌ Ошибка: {e}\n\n```\n{traceback.format_exc()}\n```",
            "", "", ""
        )

    # === OCR / ошибка OCR ===
    if result.ocr_error:
        ocr_text = f"⚠ {result.ocr_error}"
    elif result.ocr_lines:
        ocr_text = "\n".join(result.ocr_lines)
    else:
        ocr_text = "(фото не загружено)"

    # === Сводка по найденным препаратам ===
    found = [d for d in result.report.drugs if d.found]
    not_found = [d for d in result.report.drugs if not d.found and d.query]

    if found:
        lines = [f"### Найдено препаратов: {len(found)}"]
        for i, d in enumerate(found, 1):
            lines.append(f"\n**[{i}] {d.name}**")
            src_label = {'photo': 'из фото', 'manual': 'ручной ввод', 'prompt': 'из текста запроса'}.get(d.source, d.source)
            lines.append(f"- *Распознано как:* `{d.query}` ({src_label})")
            if d.substances_raw:
                lines.append(f"- *Действующие вещества:* {', '.join(d.substances_raw)}")
            if d.uses:
                lines.append(f"- *Применение:* {d.uses}")
            if d.side_effects:
                lines.append(f"- *Побочные эффекты:* {d.side_effects}")
            if d.manufacturer:
                lines.append(f"- *Производитель:* {d.manufacturer}")
        if not_found:
            lines.append(f"\n---\n⚠ **Не найдены в базе:** {', '.join(d.query for d in not_found)}")
        summary = "\n".join(lines)
    else:
        summary = (
            "🤔 **Препараты не найдены в справочнике.**\n\n"
            "Подробности — во вкладке «Совет ассистента». Попробуйте:\n"
            "- ввести международное название действующего вещества (например, `Paracetamol` вместо торговой марки),\n"
            "- проверить написание / распознанный текст во вкладке «OCR»,\n"
            "- ввести название вручную."
        )

    # === Конфликты ===
    if result.report.conflicts:
        general = [c for c in result.report.conflicts if c.get('is_general_example')]
        within = [c for c in result.report.conflicts if c.get('within_same_drug') and not c.get('is_general_example')]
        between = [c for c in result.report.conflicts if not c.get('within_same_drug') and not c.get('is_general_example')]
        lines = []

        if general:
            lines.append(f"### Примеры известных взаимодействий ({len(general)})")
            lines.append(
                "_Поскольку указан только один препарат, ниже — несколько примеров "
                "из базы: лекарств, с которыми у него есть зарегистрированные взаимодействия._\n"
            )
            for c in general:
                lines.append(f"- **{c['drug_a']} ↔ {c['drug_b']}**")
                lines.append(f"  > {c['description']}")

        if within or between:
            if general:
                lines.append("\n---\n")
            lines.append(f"### Найдено взаимодействий: {len(within) + len(between)}\n")
            if within:
                lines.append("#### Внутри одного препарата")
                for c in within:
                    lines.append(f"- **{c['drug_a']} ↔ {c['drug_b']}** (внутри *{c['source_a']}*)")
                    lines.append(f"  > {c['description']}")
            if between:
                if within:
                    lines.append("")
                lines.append("#### Между разными препаратами")
                for c in between:
                    lines.append(f"- **{c['drug_a']}** (из *{c['source_a']}*) ↔ **{c['drug_b']}** (из *{c['source_b']}*)")
                    lines.append(f"  > {c['description']}")
        conflicts_md = "\n".join(lines)
    elif found:
        conflicts_md = "✅ _Взаимодействий между указанными препаратами в базе не найдено._"
    else:
        conflicts_md = "_Препараты не найдены — проверка взаимодействий не производилась._"

    # === Совет от LLM ===
    advice = result.llm.text
    if result.llm.used_fallback:
        advice = "⚠️ **Локальная LLM недоступна — показан сырой отчёт.**\n\n" + advice

    return ocr_text, summary, conflicts_md, advice


def load_demo_scenario():
    """Возвращает случайный демо-сценарий для заполнения полей."""
    scenario = random.choice(DEMO_SCENARIOS)
    return (
        scenario['drug_list'],
        scenario['prompt'],
        f"💡 Загружен пример: **{scenario['title']}**. Нажмите «Анализировать» для запуска.",
    )


FINETUNED_LABEL = "🎓 Дообученная модель (finetuned)"


def _model_choices():
    """Список моделей: дообученная (если адаптер на месте) + модели Ollama."""
    choices = []
    try:
        from local_llm import is_available
        if is_available():
            choices.append(FINETUNED_LABEL)
    except ImportError:
        pass
    ollama = list_models(DEFAULT_HOST)
    choices.extend(ollama if ollama else [DEFAULT_MODEL])
    return choices


def refresh_models():
    choices = _model_choices()
    # По умолчанию выбираем дообученную, если она доступна
    default = FINETUNED_LABEL if FINETUNED_LABEL in choices else choices[0]
    return gr.update(choices=choices, value=default)


def build_interface():
    with gr.Blocks(title="Med Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 💊 Med Assistant\n"
            "_Локальный медицинский ассистент: фото препарата + список других лекарств → "
            "анализ всех взаимодействий → рекомендации._\n\n"
            "**Дисклеймер:** информация носит справочный характер. Финальное решение всегда принимает врач."
        )

        # Подсказка для новых пользователей
        hint = gr.Markdown(
            "👋 **Впервые здесь?** Нажмите «🎲 Пример работы» — загрузится готовый сценарий, "
            "и вы увидите, как ассистент анализирует препараты."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Препараты")
                image = gr.Image(label="Фото упаковки (опционально)", type="filepath", height=220)

                drug_list = gr.Textbox(
                    label="Список препаратов (по одному на строку или через запятую)",
                    placeholder=(
                        "Аспирин\n"
                        "Ибупрофен\n"
                        "Парацетамол\n\n"
                        "Можно перечислить и через запятую: Aspirin, Ibuprofen, Paracetamol\n"
                        "Допустимы как торговые названия (Аугментин, Augmentin 625), так и МНН (Amoxicillin)."
                    ),
                    lines=5,
                )

                gr.Markdown("### 2. Запрос пользователя")
                user_prompt = gr.Textbox(
                    label="Что вы хотите узнать?",
                    placeholder="Например: «Безопасно ли принимать эти препараты вместе?», «Какие могут быть побочки?»",
                    lines=2,
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Анализировать", variant="primary", size="lg", scale=2)
                    demo_btn = gr.Button("🎲 Пример работы", size="lg", scale=1)

                with gr.Accordion("⚙️ Настройки", open=False):
                    _initial_choices = _model_choices()
                    model_name = gr.Dropdown(
                        label="Модель",
                        choices=_initial_choices,
                        value=_initial_choices[0],
                        allow_custom_value=True,
                    )
                    refresh_btn = gr.Button("🔄 Обновить список моделей", size="sm")
                    languages = gr.CheckboxGroup(
                        label="Языки OCR (для распознавания фото)",
                        choices=["en", "ru"],
                        value=["en", "ru"],
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Результат")
                with gr.Tab("💬 Совет ассистента"):
                    advice_out = gr.Markdown()
                with gr.Tab("📋 Препараты"):
                    summary_out = gr.Markdown()
                with gr.Tab("⚠️ Взаимодействия"):
                    conflicts_out = gr.Markdown()
                with gr.Tab("🔍 OCR"):
                    ocr_out = gr.Textbox(label="Распознанный текст", lines=8, interactive=False)

        analyze_btn.click(
            fn=analyze_handler,
            inputs=[image, user_prompt, drug_list, model_name, languages],
            outputs=[ocr_out, summary_out, conflicts_out, advice_out],
        )
        demo_btn.click(
            fn=load_demo_scenario,
            outputs=[drug_list, user_prompt, hint],
        )
        refresh_btn.click(fn=refresh_models, outputs=model_name)

        gr.Markdown(
            "---\n"
            "**Как это работает:** программа находит препараты в справочнике (~11 800 записей), "
            "извлекает их состав, и затем перебирает **все пары действующих веществ** "
            "(включая вещества внутри одного комбинированного препарата) для поиска взаимодействий "
            "в базе ~191 000 пар. LLM (Ollama: Llama 3, Mistral, Qwen) затем "
            "пересказывает найденные факты человеческим языком.\n\n"
            "💡 Для полноценной работы запустите [Ollama](https://ollama.com) "
            "с моделью `llama3:8b`. Без неё вы получите сырой технический отчёт."
        )

    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
