"""
Этап 4: Интеграция с локальной LLM через Ollama
- Используем встроенный HTTP-API Ollama (POST /api/generate),
  чтобы не зависеть от пакета `ollama` (его может не быть установлен).
- Поддерживаем стриминг и обычный режим.
- Если Ollama недоступен - возвращаем "сырой" отчёт с предупреждением,
  чтобы приложение не падало в демо-режиме.

Запускайте Ollama локально:
    ollama serve
    ollama pull llama3:8b   # или mistral, или qwen2.5:7b
"""
from __future__ import annotations
import json
import urllib.request
import urllib.error
from dataclasses import dataclass


# Модель Ollama по умолчанию. После дообучения (см. finetune/) и экспорта
# в Ollama здесь можно указать "med-assistant" — обученную под задачу версию.
DEFAULT_MODEL = "llama3:8b"
DEFAULT_HOST = "http://localhost:11434"
REQUEST_TIMEOUT = 120  # секунд


SYSTEM_PROMPT_RU = """Ты — русскоязычный медицинский ассистент-консультант. Твоя задача — на основе ТОЛЬКО предоставленных данных
объяснить пользователю простым языком:
1. Что это за препарат и для чего он применяется.
2. Какие у него побочные эффекты.
3. Взаимодействия с другими лекарствами — насколько они опасны и что делать.
4. Рекомендации по применению.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
- ВЕСЬ ОТВЕТ ДОЛЖЕН БЫТЬ НА РУССКОМ ЯЗЫКЕ. Не используй английские слова и фразы — переводи всё.
- Названия лекарств (Aspirin, Ibuprofen, Paracetamol) можно оставлять как есть — в скобках можно дать русский эквивалент.
- Все остальные английские термины (применение, побочные эффекты, симптомы, описания взаимодействий) ОБЯЗАТЕЛЬНО переводи на русский.
- Не выдумывай факты. Если в данных чего-то нет — так и скажи.
- Не назначай дозировки и не заменяй врача. Финальное решение — за лечащим врачом.
- Будь кратким и структурированным: короткие абзацы или маркированные списки.

СЛОВАРЬ ЧАСТЫХ ТЕРМИНОВ (используй при переводе данных из отчёта):
- Treatment of Bacterial infections → Лечение бактериальных инфекций
- Pain relief → Облегчение боли
- Inflammation → Воспаление
- Fever → Жар, лихорадка
- Hypertension → Артериальная гипертензия (повышенное давление)
- Diabetes → Сахарный диабет
- Heart attack → Инфаркт миокарда
- Stroke → Инсульт
- Gout → Подагра
- Cancer → Онкологические заболевания
- Vomiting → Рвота
- Nausea → Тошнота
- Diarrhea → Диарея (понос)
- Headache → Головная боль
- Dizziness → Головокружение
- Drowsiness, Sleepiness → Сонливость
- Fatigue, Tiredness → Усталость, утомляемость
- Rash → Сыпь
- Itching → Зуд
- Bleeding → Кровотечение
- Indigestion → Расстройство пищеварения
- Loss of appetite → Потеря аппетита
- Abdominal pain → Боль в животе
- "increase the serum concentration" → "повысить концентрацию в крови"
- "decrease the therapeutic efficacy" → "снизить терапевтическую эффективность"
- "increase the risk of adverse effects" → "повысить риск побочных эффектов"
- "may decrease the excretion rate" → "может снизить скорость выведения"

ПРИМЕР ОТВЕТА:
Пользователь спросил: «Можно ли принимать Аспирин с Ибупрофеном?»
Отчёт: Препарат Aspirin (Acetylsalicylic acid). Применение: Prevention of heart attack. Конфликт: Acetylsalicylic acid + Ibuprofen — increased risk of adverse effects.

Правильный ответ:
**Аспирин (ацетилсалициловая кислота)** применяется для профилактики инфаркта миокарда.

⚠ **Внимание: совместный приём с ибупрофеном повышает риск побочных эффектов.** Ибупрофен может снижать защитное действие аспирина на сердечно-сосудистую систему, а также повышать риск желудочно-кишечных кровотечений.

**Рекомендации:** избегать совместного приёма; если необходимо принимать оба препарата, ибупрофен лучше принимать минимум через 8 часов после аспирина.

⚕ Обязательно проконсультируйтесь с врачом — самостоятельная корректировка терапии опасна.
"""

SYSTEM_PROMPT_EN = """You are a medical assistant. Based ONLY on the provided data from medical references,
explain to the user in plain language:
1. What the drug is and what it's used for.
2. Its side effects.
3. If there are interactions with other drugs - how serious they are and what to do.
4. Recommendations for use.

IMPORTANT RULES:
- Do not invent facts. If something isn't in the data, say so.
- Do not prescribe dosages or replace a doctor. Always remind the user that the final decision rests with their physician.
- ALWAYS RESPOND IN ENGLISH.
- Be concise and structured: use short paragraphs or bullet lists.
"""


def detect_language(text: str) -> str:
    """Простой детектор: если в тексте есть кириллица - русский, иначе английский."""
    if not text:
        return 'en'
    import re as _re
    cyrillic = len(_re.findall(r'[\u0400-\u04FF]', text))
    latin = len(_re.findall(r'[a-zA-Z]', text))
    if cyrillic > latin / 2:  # хотя бы треть кириллицы -> русский
        return 'ru'
    return 'en'


def build_user_prompt(report_text: str, language: str = 'ru') -> str:
    """Формирует пользовательский промпт по шаблону из плана (Этап 4)."""
    if language == 'ru':
        return f"""Ниже — структурированный технический отчёт из медицинских баз. На его основе:
объясни пользователю простым языком возможные риски, побочные эффекты и дай рекомендации
по применению. Если найдены взаимодействия с другими препаратами — обязательно их проговори.
ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.

=== ТЕХНИЧЕСКИЙ ОТЧЁТ ===
{report_text}
=== КОНЕЦ ОТЧЁТА ===

Ответ построй по структуре:
1. Краткое описание препарата.
2. Возможные побочные эффекты (на что обратить внимание).
3. Взаимодействия с другими препаратами (если найдены) — насколько серьёзны и что делать.
4. Рекомендации по применению.
5. Дисклеймер о консультации с врачом."""
    else:
        return f"""Below is a structured technical report from medical references. Based on it:
explain to the user in simple language the possible risks, side effects, and provide recommendations
for use. If interactions with other drugs are found — be sure to mention them.
RESPOND ONLY IN ENGLISH.

=== TECHNICAL REPORT ===
{report_text}
=== END OF REPORT ===

Structure your answer as:
1. Brief description of the drug.
2. Possible side effects (what to watch for).
3. Drug interactions (if any found) - how serious and what to do.
4. Recommendations for use.
5. Disclaimer about consulting a physician."""


@dataclass
class LLMResponse:
    text: str
    used_fallback: bool
    error: str | None = None


def _ollama_request(prompt: str, system: str, model: str, host: str,
                    temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return obj.get("response", "").strip()


def list_models(host: str = DEFAULT_HOST) -> list[str]:
    """Возвращает список локально установленных моделей Ollama (или []).
    Используется в GUI для выпадающего списка."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in obj.get("models", []) if m.get("name")]
    except Exception:
        return []


def generate_advice(report_text: str,
                    user_prompt_text: str = '',
                    model: str = DEFAULT_MODEL,
                    host: str = DEFAULT_HOST,
                    temperature: float = 0.3,
                    language: str | None = None) -> LLMResponse:
    """Генерирует совет на основе технического отчёта.
    Язык определяется автоматически по `user_prompt_text`, если language не задан.
    Для русского — предварительно переводит медицинские термины в отчёте, чтобы
    помочь модели сформировать русскоязычный ответ.
    При недоступности Ollama возвращает читаемый fallback."""
    if language is None:
        language = detect_language(user_prompt_text or report_text)

    # Предпереводим медтермины - это сильно стабилизирует русский ответ Llama 3 8B
    if language == 'ru':
        try:
            from medical_glossary import translate_medical_terms
            report_text = translate_medical_terms(report_text)
        except ImportError:
            pass

    system_prompt = SYSTEM_PROMPT_RU if language == 'ru' else SYSTEM_PROMPT_EN
    user_prompt = build_user_prompt(report_text, language=language)

    try:
        text = _ollama_request(user_prompt, system_prompt, model, host, temperature)
        if not text:
            return LLMResponse(
                text=_fallback_text(report_text, "LLM вернул пустой ответ", language),
                used_fallback=True,
                error="Empty response from LLM",
            )

        # Постобработка: если ответ должен быть на русском, но содержит много
        # английского - прогоняем через глоссарий ещё раз (страховка).
        if language == 'ru':
            try:
                from medical_glossary import translate_medical_terms
                # Считаем, сколько латиницы в ответе
                import re as _re
                latin_chars = len(_re.findall(r'[A-Za-z]', text))
                cyrillic_chars = len(_re.findall(r'[\u0400-\u04FF]', text))
                if cyrillic_chars > 0 and latin_chars / max(cyrillic_chars, 1) > 0.15:
                    text = translate_medical_terms(text)
            except ImportError:
                pass

        return LLMResponse(text=text, used_fallback=False)
    except urllib.error.URLError as e:
        return LLMResponse(
            text=_fallback_text(report_text,
                                f"Не удалось подключиться к Ollama ({host}). "
                                f"Запустите 'ollama serve' и убедитесь, что модель '{model}' установлена.",
                                language),
            used_fallback=True,
            error=str(e),
        )
    except Exception as e:
        return LLMResponse(
            text=_fallback_text(report_text, f"Ошибка LLM: {e}", language),
            used_fallback=True,
            error=str(e),
        )


def _fallback_text(report_text: str, reason: str, language: str = 'ru') -> str:
    """Если LLM недоступен - выдаём аккуратный отчёт без «красивого» текста,
    но со всеми найденными фактами. Никаких галлюцинаций."""
    if language == 'ru':
        return (
            "⚠️ Локальная модель недоступна, поэтому ниже — сырые данные из справочников.\n"
            f"Причина: {reason}\n\n"
            "═══════════════════════════════════════\n"
            f"{report_text}\n"
            "═══════════════════════════════════════\n\n"
            "💡 Для полноценного ответа запустите Ollama локально:\n"
            "   1) ollama serve\n"
            "   2) ollama pull llama3:8b\n"
            "   3) перезапустите приложение."
        )
    return (
        "⚠️ Local model is unavailable, so below is raw data from references.\n"
        f"Reason: {reason}\n\n"
        "═══════════════════════════════════════\n"
        f"{report_text}\n"
        "═══════════════════════════════════════\n\n"
        "💡 To get a full answer, start Ollama locally:\n"
        "   1) ollama serve\n"
        "   2) ollama pull llama3:8b\n"
        "   3) restart the application."
    )


if __name__ == '__main__':
    print("Доступные модели Ollama:", list_models())
    fake_report = (
        "Препарат: Augmentin 625 Duo Tablet\n"
        "Действующие вещества: Amoxycillin, Clavulanic Acid\n"
        "Применение: Treatment of Bacterial infections\n"
        "Побочные эффекты: Vomiting Nausea Diarrhea\n"
        "Найденные взаимодействия (1):\n"
        "  • Amoxicillin ↔ Probenecid: The serum concentration of Probenecid can be increased.\n"
        "Запрос пользователя: можно ли пить вместе с пробенецидом?"
    )
    user_q = "можно ли пить вместе с пробенецидом?"
    resp = generate_advice(fake_report, user_prompt_text=user_q)
    print(f"Detected language: {detect_language(user_q)}")
    print(resp.text[:500])
    print("---")
    print(f"used_fallback: {resp.used_fallback}, error: {resp.error}")
