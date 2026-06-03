"""
Этап 2: Модуль ввода и распознавания (Vision & Input)
- EasyOCR с английским и русским языками
- Извлечение названия препарата с упаковки
- Сопоставление распознанного текста с известными названиями в базе (fuzzy)
"""
from typing import Optional
import re

try:
    from rapidfuzz import process, fuzz
    _FUZZ_OK = True
except ImportError:
    try:
        from thefuzz import process, fuzz
        _FUZZ_OK = True
    except ImportError:
        import difflib
        _FUZZ_OK = False


# Слова, которые часто на упаковках, но не являются названиями препаратов
NOISE_WORDS = {
    'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'injection',
    'cream', 'ointment', 'gel', 'drops', 'spray', 'mg', 'ml', 'mcg',
    'expiry', 'mfg', 'batch', 'rx', 'prescription', 'dosage',
    'таблетки', 'таблетка', 'капсулы', 'капсула', 'сироп', 'мг', 'мл',
    'годен', 'до', 'серия', 'рецепт', 'упаковка', 'инструкция',
}


_reader = None


def get_reader(languages=('en', 'ru'), gpu: bool = False):
    """Ленивая инициализация EasyOCR (загружает модели при первом вызове)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
    return _reader


def extract_text(image_path: str, languages=('en', 'ru'), gpu: bool = False) -> list[str]:
    """
    Распознаёт текст с упаковки. Возвращает список строк (по строкам с уверенностью >= 0.3).
    """
    reader = get_reader(languages=languages, gpu=gpu)
    # detail=1 -> [bbox, text, conf]
    raw = reader.readtext(image_path, detail=1, paragraph=False)
    lines = []
    for item in raw:
        # rapidfuzz/easyocr возвращают tuple, а некоторые версии - list
        if len(item) >= 3:
            _, text, conf = item[0], item[1], item[2]
            if conf >= 0.3 and text.strip():
                lines.append(text.strip())
    return lines


def _clean_token(token: str) -> str:
    """Чистит токен от мусорных символов, оставляет буквы/цифры/пробелы/дефисы."""
    return re.sub(r'[^\w\s\-]', ' ', token, flags=re.UNICODE).strip()


def _is_noise(token: str) -> bool:
    t = token.lower().strip()
    if not t or len(t) < 3:
        return True
    if t in NOISE_WORDS:
        return True
    if re.fullmatch(r'\d+', t):  # просто число
        return True
    return False


def _drug_name_likelihood_score(text: str) -> int:
    """
    Эвристика: насколько строка «похожа на название препарата».
    Чем больше - тем выше приоритет кандидата.

    Названия препаратов на упаковках обычно:
      - КОРОТКИЕ (1-3 слова, типа "Augmentin 625 Duo")
      - содержат буквы и иногда цифры (дозировка)
      - не состоят из слов-индикаторов описания (each, contains, equivalent, mfg, exp...)
      - могут быть в верхнем регистре или иметь заглавную первую букву
    """
    if not text:
        return 0
    score = 0
    s = text.strip()
    sl = s.lower()
    words = s.split()
    n_words = len(words)

    # Длина 4..25 символов - типичное название
    if 4 <= len(s) <= 25:
        score += 15
    elif 26 <= len(s) <= 40:
        score += 5
    else:
        score -= 10  # слишком длинная

    # Количество слов: 1-3 - идеально, 4 - средне, 5+ - почти точно описание
    if n_words == 1:
        score += 10
    elif 2 <= n_words <= 3:
        score += 12
    elif n_words == 4:
        score += 3
    else:
        score -= 10  # 5+ слов - это, вероятно, фраза-описание

    # Жёсткий минус за слова-индикаторы описания
    DESCRIPTION_WORDS = {
        'each', 'contains', 'equivalent', 'composition', 'ingredients',
        'storage', 'mfg', 'mfd', 'exp', 'expiry', 'batch', 'mrp',
        'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'cream',
        'film', 'coated', 'sustained', 'release', 'extended',
        'pharmaceuticals', 'limited', 'pharma', 'laboratories', 'industries',
        'каждая', 'содержит', 'состав', 'хранение', 'годен',
        'таблетка', 'таблетки', 'капсула', 'капсулы',
    }
    desc_hits = sum(1 for w in words if w.lower() in DESCRIPTION_WORDS)
    score -= desc_hits * 12

    # Соотношение букв к не-буквам
    letters = sum(1 for c in s if c.isalpha())
    if letters >= 4:
        score += min(letters // 3, 10)

    # Цифры - обычно дозировка, нормально (но если число большое - это уже не дозировка)
    digits = sum(1 for c in s if c.isdigit())
    if 0 < digits <= 4:
        score += 4

    # Капитализация
    if s and s[0].isupper():
        score += 3
    # CAPS LOCK style (как часто пишут названия на упаковках)
    if s.isupper() and len(s) <= 20:
        score += 5

    # Минус если все слова - служебные/мусорные
    non_noise = [w for w in words if not _is_noise(w)]
    if not non_noise:
        score -= 100

    return score


def find_medicine_name(ocr_lines: list[str],
                       medicine_index: dict,
                       score_cutoff: int = 85) -> Optional[dict]:
    """
    По распознанным строкам ищет лучшее совпадение в medicine_index.
    Стратегия:
      1. Перевод RU→EN для строк, содержащих кириллицу.
      2. Объединяем все непустые строки -> кандидаты (отдельные строки + пары соседних).
      3. Для каждого кандидата ищем fuzzy-совпадение по ключам индекса.
      4. Возвращаем запись с наибольшим скором, если он >= score_cutoff.

    Возвращает: запись из medicine_index с добавленным полем 'match_score',
    либо None, если хорошего совпадения нет.
    """
    if not ocr_lines or not medicine_index:
        return None

    # Расширяем строки переводом RU->EN, если есть кириллица
    expanded_lines = list(ocr_lines)
    try:
        from synonyms import translate_ru_to_en
        for line in ocr_lines:
            if re.search(r'[\u0400-\u04FF]', line):
                translated, _ = translate_ru_to_en(line)
                if translated != line:
                    expanded_lines.append(translated)
    except ImportError:
        pass

    # Готовим кандидатов
    candidates = []
    for line in expanded_lines:
        cleaned = _clean_token(line)
        if cleaned and not _is_noise(cleaned):
            candidates.append(cleaned)

    # Соседние строки часто разбивают название: "Augmentin 625" + "Duo Tablet"
    # Склеиваем пары и тройки соседних строк.
    for i in range(len(expanded_lines) - 1):
        merged = _clean_token(f"{expanded_lines[i]} {expanded_lines[i+1]}")
        if merged:
            candidates.append(merged)
    for i in range(len(expanded_lines) - 2):
        merged3 = _clean_token(f"{expanded_lines[i]} {expanded_lines[i+1]} {expanded_lines[i+2]}")
        if merged3:
            candidates.append(merged3)

    # Дедупликация с сохранением порядка
    seen = set()
    deduped = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            deduped.append(c)
    candidates = deduped

    if not candidates:
        return None

    keys = list(medicine_index.keys())  # ключи уже в lower
    best_record = None
    best_score = 0
    best_query = ''

    for cand in candidates:
        cand_lower = cand.lower()
        if _FUZZ_OK:
            result = process.extractOne(cand_lower, keys, scorer=fuzz.WRatio)
            if result is None:
                continue
            match_key, score = result[0], result[1]
        else:
            import difflib
            matches = difflib.get_close_matches(cand_lower, keys, n=1, cutoff=0.5)
            if not matches:
                continue
            match_key = matches[0]
            score = int(difflib.SequenceMatcher(None, cand_lower, match_key).ratio() * 100)

        if score > best_score:
            best_score = score
            best_record = medicine_index[match_key]
            best_query = cand  # сохраняем именно тот кандидат, что сматчился

    # Двойная проверка: помимо общего скора, первое слово запроса должно
    # хотя бы частично присутствовать в найденном названии. Это отсекает
    # ложные срабатывания вроде "Aspirin" -> "Asthalin Tablet".
    if best_record and best_score >= score_cutoff:
        if not _validate_match(best_query.lower(), best_record['original_name']):
            return None
        # Возвращаем запись + match_score + matched_query (тот фрагмент OCR, что сматчился)
        return {**best_record, 'match_score': best_score, 'matched_query': best_query}
    return None


def _validate_match(query_lower: str, found_name: str) -> bool:
    """
    Дополнительная проверка качества совпадения.
    Главное правило: первый значимый токен запроса должен встречаться
    в найденном названии (точно или с очень высокой схожестью).
    """
    query_tokens = [t for t in query_lower.split() if len(t) >= 3 and not _is_noise(t)]
    if not query_tokens:
        return True  # ничего проверять
    first = query_tokens[0]
    found_lower = found_name.lower()

    # 1) Префиксное вхождение
    if first in found_lower:
        return True

    # 2) Хотя бы один токен из запроса должен быть в found
    for t in query_tokens:
        if t in found_lower:
            return True

    # 3) Любой токен из found похож на токен из запроса >= 85%
    found_tokens = [t for t in re.split(r'\s+', found_lower) if len(t) >= 3]
    if _FUZZ_OK:
        for qt in query_tokens:
            for ft in found_tokens:
                if fuzz.WRatio(qt, ft) >= 85:
                    return True
    else:
        import difflib
        for qt in query_tokens:
            for ft in found_tokens:
                if difflib.SequenceMatcher(None, qt, ft).ratio() >= 0.85:
                    return True
    return False


def extract_additional_drugs(user_prompt: str, medicine_index: dict,
                             alias_map: dict, top_k: int = 5) -> list[str]:
    """
    Из текста пользователя ('Принимаю это вместе с Аспирином и Ибупрофеном')
    извлекает упомянутые препараты или вещества.
    Возвращает канонические имена веществ (как в db_drug_interactions).
    """
    if not user_prompt or not user_prompt.strip():
        return []

    # Сначала переводим русские названия в английские (если они есть в словаре)
    try:
        from synonyms import translate_ru_to_en
        translated_prompt, _ = translate_ru_to_en(user_prompt)
    except ImportError:
        translated_prompt = user_prompt

    # Токенизация: слова длиной >= 3 (берём из переведённой версии)
    tokens = re.findall(r'[\w\-]{3,}', translated_prompt, flags=re.UNICODE)
    # Также добавляем токены из оригинала - на случай, если что-то осталось на русском
    tokens += re.findall(r'[\w\-]{3,}', user_prompt, flags=re.UNICODE)
    tokens = [t for t in tokens if not _is_noise(t)]

    found_canonical = []
    seen = set()

    for tok in tokens:
        tok_lower = tok.lower()

        # 1. Ищем в alias_map напрямую (вещества)
        if tok_lower in alias_map:
            canon = alias_map[tok_lower]
            if canon.lower() not in seen:
                found_canonical.append(canon)
                seen.add(canon.lower())
            continue

        # 2. Fuzzy ищем в alias_map
        if _FUZZ_OK:
            result = process.extractOne(tok_lower, list(alias_map.keys()),
                                        scorer=fuzz.WRatio, score_cutoff=85)
            if result:
                canon = alias_map[result[0]]
                if canon.lower() not in seen:
                    found_canonical.append(canon)
                    seen.add(canon.lower())
                continue

        # 3. Может быть, пользователь назвал торговое имя -> ищем в medicine_index
        if _FUZZ_OK:
            result = process.extractOne(tok_lower, list(medicine_index.keys()),
                                        scorer=fuzz.WRatio, score_cutoff=88)
            if result:
                rec = medicine_index[result[0]]
                for canon in rec['substances_canonical']:
                    if canon.lower() not in seen:
                        found_canonical.append(canon)
                        seen.add(canon.lower())

    return found_canonical[:top_k]


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_preparation import load_cache

    base = Path(__file__).resolve().parent.parent
    medicine_index, _, alias_map = load_cache(str(base / 'cache'))

    # Демо без реального изображения - имитируем OCR
    fake_ocr = ["Augmentin 625", "Duo Tablet", "GSK"]
    result = find_medicine_name(fake_ocr, medicine_index)
    print("Имитация OCR:", fake_ocr)
    print("Найдено:", result['original_name'] if result else None)
    print("Состав:", result['substances_raw'] if result else None)

    # Тест парсинга промпта
    prompt = "Принимаю вместе с Aspirin и Ibuprofen"
    drugs = extract_additional_drugs(prompt, medicine_index, alias_map)
    print(f"\nПромпт: {prompt}")
    print(f"Извлечённые вещества: {drugs}")
