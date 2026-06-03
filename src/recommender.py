"""
Этап 5: Логика рекомендаций (оркестрация)

Главный метод - analyze_multi(): принимает список запросов на препараты + текст.
Каждый запрос превращается в DrugEntry и проходит цепочку:
  торговое название → если не нашли → название вещества → если не нашли → not found.

Старый метод analyze() сохранён для совместимости.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from data_preparation import load_cache
from ocr_module import extract_text, find_medicine_name, _validate_match
from search_engine import build_report, format_report_for_llm, TechnicalReport, DrugEntry
from llm_client import generate_advice, LLMResponse, DEFAULT_MODEL, detect_language


@dataclass
class AssistantResult:
    ocr_lines: list[str] = field(default_factory=list)
    ocr_error: str | None = None
    report: TechnicalReport = field(default_factory=TechnicalReport)
    report_text: str = ''
    llm: LLMResponse = field(default_factory=lambda: LLMResponse(text='', used_fallback=False))


class MedAssistant:
    """Главный объект приложения. Кэширует индексы, чтобы не загружать на каждый запрос."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = str(cache_dir)
        self.medicine_index, self.interaction_index, self.alias_map = load_cache(self.cache_dir)

    def _safe_ocr(self, image_path: str,
                  ocr_languages: tuple[str, ...],
                  use_gpu: bool) -> tuple[list[str], str | None]:
        """
        Безопасный OCR: сначала валидирует/нормализует изображение через Pillow
        (что покрывает HEIC через pillow-heif, BMP, TIFF, GIF, кириллические пути).
        Возвращает (lines, error_message). Если error_message задан - OCR не удался.
        """
        from pathlib import Path as _Path
        p = _Path(image_path)
        if not p.exists() or not p.is_file():
            return [], "Файл изображения не найден или путь некорректен."
        if p.stat().st_size == 0:
            return [], "Файл изображения пустой."

        # ВАЖНО: открываем через Pillow, а не отдаём путь напрямую в EasyOCR.
        # Иначе cv2.imread внутри EasyOCR падает на кириллических путях и HEIC.
        try:
            from PIL import Image, ImageOps
            import numpy as np
        except ImportError:
            try:
                lines = extract_text(image_path, languages=ocr_languages, gpu=use_gpu)
                return lines, None
            except Exception as e:
                return [], self._format_ocr_error(e)

        try:
            with Image.open(image_path) as img:
                # NB: фото с телефона часто перевёрнуты — выправляем по EXIF до OCR
                img = ImageOps.exif_transpose(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                arr = np.array(img)
        except Exception as e:
            return [], (
                "Не удалось прочитать изображение. "
                "Возможно, файл повреждён или используется неподдерживаемый формат "
                "(попробуйте PNG или JPEG). Попробуйте другое фото."
            )

        try:
            from ocr_module import get_reader
            reader = get_reader(languages=ocr_languages, gpu=use_gpu)
            raw = reader.readtext(arr, detail=1, paragraph=False)
            lines = []
            for item in raw:
                if len(item) >= 3:
                    _, text, conf = item[0], item[1], item[2]
                    if conf >= 0.3 and text.strip():  # отсекаем низкоуверенный мусор
                        lines.append(text.strip())
            if not lines:
                return [], (
                    "Не удалось распознать текст с фото. "
                    "Попробуйте сделать снимок крупнее, при хорошем освещении, "
                    "так чтобы название препарата было читаемо."
                )
            return lines, None
        except Exception as e:
            return [], self._format_ocr_error(e)

    @staticmethod
    def _format_ocr_error(exc: Exception) -> str:
        """Превращает технические ошибки OCR в понятное сообщение пользователю."""
        msg = str(exc)
        # ВАЖНО: это типовая ошибка EasyOCR, когда cv2.imread не смог прочитать файл
        if 'NoneType' in msg and 'shape' in msg:
            return (
                "Не удалось прочитать файл изображения. "
                "Возможно, формат не поддерживается или путь содержит специальные символы. "
                "Попробуйте загрузить фото в формате JPEG или PNG."
            )
        if 'cannot identify image' in msg.lower():
            return "Файл не распознан как изображение. Попробуйте другое фото."
        if 'memory' in msg.lower():
            return "Изображение слишком большое. Попробуйте уменьшить разрешение."
        return f"Не удалось обработать фото. Попробуйте другой снимок. (Технические детали: {msg[:100]})"

    @staticmethod
    def _pick_best_name_guess(ocr_lines: list[str]) -> str:
        """Если препарат не нашёлся, берём самую похожую на название строку
        (вместо того чтобы передавать весь текст с упаковки)."""
        from ocr_module import _drug_name_likelihood_score, _clean_token, _is_noise
        candidates = []
        for line in ocr_lines:
            cleaned = _clean_token(line)
            if cleaned and not _is_noise(cleaned):
                candidates.append(cleaned)
        if not candidates:
            return (ocr_lines[0] if ocr_lines else '')[:60]
        candidates.sort(key=_drug_name_likelihood_score, reverse=True)
        return candidates[0][:60]

    # ============================================================
    # Поиск одного препарата по строке-запросу
    # ============================================================
    def resolve_one(self, query: str, source: str = 'manual') -> DrugEntry:
        """
        Превращает строку (как ввёл пользователь) в DrugEntry.
        Каскад поиска: торговое название → вещество в Medicine_Details →
        вещество только в базе взаимодействий → не найдено.
        """
        entry = DrugEntry(source=source, query=query.strip(), found=False)
        if not query or not query.strip():
            return entry

        en_query = query.lower().strip()
        try:
            from synonyms import translate_ru_to_en
            translated, _ = translate_ru_to_en(en_query)
            if translated != en_query:
                en_query = translated
        except ImportError:
            pass

        rec = self._find_as_trade_name(en_query)
        if rec is None:
            rec = self._find_by_substance(en_query)

        # КЛЮЧЕВОЙ кейс: вещество есть в базе взаимодействий, но не в Medicine_Details.
        # Без этой ветки препараты вроде Diazepam/Warfarin не находились бы вообще.
        if rec is None:
            canonical = self._find_in_interactions_only(en_query)
            if canonical:
                rec = {
                    'original_name': f"{canonical} (только в базе взаимодействий)",
                    'substances_raw': [canonical],
                    'substances_canonical': [canonical],
                    'uses': '(нет в справочнике препаратов - данные только о взаимодействиях)',
                    'side_effects': '(нет в справочнике препаратов)',
                    'manufacturer': '',
                }

        if rec is None:
            return entry

        entry.found = True
        entry.name = rec['original_name']
        entry.substances_raw = rec['substances_raw']
        entry.substances_canonical = rec['substances_canonical']
        entry.uses = rec['uses']
        entry.side_effects = rec['side_effects']
        entry.manufacturer = rec['manufacturer']
        return entry

    def _find_in_interactions_only(self, query: str) -> str | None:
        """Ищет вещество прямо в interaction_index, если его нет в Medicine_Details.
        Возвращает каноническое имя (как оно записано в interactions) или None."""
        from synonyms import normalize_substance_name, expand_with_synonyms

        # Строим кэш веществ один раз при первом вызове (ленивая инициализация)
        if not hasattr(self, '_interaction_drug_set'):
            drugs = set()
            for (a, b) in self.interaction_index.keys():
                drugs.add(a)
                drugs.add(b)
            self._interaction_drug_set = drugs
            self._interaction_drug_list = list(drugs)
            # ключи индекса в нижнем регистре; для показа восстанавливаем через .title()
            self._interaction_canonical = {}
            for (a, b), desc in self.interaction_index.items():
                if a not in self._interaction_canonical:
                    self._interaction_canonical[a] = a.title()
                if b not in self._interaction_canonical:
                    self._interaction_canonical[b] = b.title()

        q = query.lower().strip()
        if q in self._interaction_drug_set:
            return self._interaction_canonical.get(q, q.title())

        q_norm = normalize_substance_name(q)
        if q_norm in self._interaction_drug_set:
            return self._interaction_canonical.get(q_norm, q_norm.title())

        for variant in expand_with_synonyms(q):
            if variant in self._interaction_drug_set:
                return self._interaction_canonical.get(variant, variant.title())

        # ОСТОРОЖНО: строгий порог 92 + проверка длины — иначе ловит ложные пары
        # (напр. 'Aspirinex' → 'aspirin'). Не снижать без перепроверки на тестах.
        try:
            from rapidfuzz import process as _p, fuzz as _f
            match = _p.extractOne(q, self._interaction_drug_list,
                                   scorer=_f.WRatio, score_cutoff=92)
            if match:
                key = match[0]
                len_diff = abs(len(q) - len(key)) / max(len(q), len(key))
                if len_diff <= 0.2:
                    return self._interaction_canonical.get(key, key.title())
        except ImportError:
            try:
                from thefuzz import process as _p, fuzz as _f
                match = _p.extractOne(q, self._interaction_drug_list, scorer=_f.WRatio)
                if match and match[1] >= 92:
                    key = match[0]
                    len_diff = abs(len(q) - len(key)) / max(len(q), len(key))
                    if len_diff <= 0.2:
                        return self._interaction_canonical.get(key, key.title())
            except ImportError:
                import difflib
                m = difflib.get_close_matches(q, self._interaction_drug_list, n=1, cutoff=0.92)
                if m:
                    key = m[0]
                    len_diff = abs(len(q) - len(key)) / max(len(q), len(key))
                    if len_diff <= 0.2:
                        return self._interaction_canonical.get(key, key.title())
        return None

    def _find_as_trade_name(self, query: str) -> dict | None:
        """Ищет в medicine_index по торговому названию.
        Учитывает префиксное совпадение: 'Augmentin 625' префикс к 'Augmentin 625 Duo Tablet'."""
        keys = list(self.medicine_index.keys())
        query = query.strip().lower()

        # 1) Префиксное совпадение - наивысший приоритет
        prefix_matches = [k for k in keys if k.startswith(query)]
        if prefix_matches:
            # Берём кратчайший (самый специфичный)
            prefix_matches.sort(key=len)
            return self.medicine_index[prefix_matches[0]]

        # 2) Подстрока: запрос содержится в названии
        substring_matches = [k for k in keys if query in k]
        if substring_matches:
            substring_matches.sort(key=len)
            best = substring_matches[0]
            # Дополнительная проверка: запрос должен быть значительной частью названия
            if len(query) >= 0.5 * len(best):
                return self.medicine_index[best]

        # 3) Fuzzy
        best_match = None
        best_score = 0
        try:
            from rapidfuzz import process as _p, fuzz as _f
            match = _p.extractOne(query, keys, scorer=_f.WRatio)
            if match:
                best_match, best_score = match[0], match[1]
        except ImportError:
            try:
                from thefuzz import process as _p, fuzz as _f
                match = _p.extractOne(query, keys, scorer=_f.WRatio)
                if match:
                    best_match, best_score = match[0], match[1]
            except ImportError:
                import difflib
                m = difflib.get_close_matches(query, keys, n=1, cutoff=0.6)
                if m:
                    best_match = m[0]
                    best_score = int(difflib.SequenceMatcher(None, query, best_match).ratio() * 100)

        if (best_match and best_score >= 85
                and _validate_match(query, self.medicine_index[best_match]['original_name'])):
            return self.medicine_index[best_match]
        return None

    def _find_by_substance(self, query: str) -> dict | None:
        """Если ввели название вещества (Aspirin, Paracetamol) - ищем в alias_map."""
        from synonyms import normalize_substance_name, expand_with_synonyms

        query_norm = normalize_substance_name(query)
        canonical = None

        if query.lower() in self.alias_map:
            canonical = self.alias_map[query.lower()]
        elif query_norm in self.alias_map:
            canonical = self.alias_map[query_norm]
        else:
            for variant in expand_with_synonyms(query):
                if variant in self.alias_map:
                    canonical = self.alias_map[variant]
                    break

        if canonical is None:
            best_key, best_score = None, 0
            try:
                from rapidfuzz import process as _p, fuzz as _f
                match = _p.extractOne(query.lower(), list(self.alias_map.keys()),
                                       scorer=_f.WRatio)
                if match:
                    best_key, best_score = match[0], match[1]
            except ImportError:
                try:
                    from thefuzz import process as _p, fuzz as _f
                    match = _p.extractOne(query.lower(), list(self.alias_map.keys()),
                                           scorer=_f.WRatio)
                    if match:
                        best_key, best_score = match[0], match[1]
                except ImportError:
                    import difflib
                    m = difflib.get_close_matches(query.lower(), list(self.alias_map.keys()),
                                                  n=1, cutoff=0.92)
                    if m:
                        best_key = m[0]
                        best_score = int(difflib.SequenceMatcher(None, query.lower(), m[0]).ratio() * 100)

            if best_key and best_score >= 92:
                len_diff = abs(len(query) - len(best_key)) / max(len(query), len(best_key))
                if len_diff <= 0.2:
                    canonical = self.alias_map[best_key]

        if canonical is None:
            return None

        # Подбираем препарат с этим веществом для метаданных (применение, побочки)
        candidates = [r for r in self.medicine_index.values()
                      if canonical in r['substances_canonical']]
        if not candidates:
            return {
                'original_name': canonical,
                'substances_raw': [canonical],
                'substances_canonical': [canonical],
                'uses': '(определяется веществом, конкретный препарат не выбран)',
                'side_effects': '(зависит от препарата)',
                'manufacturer': '',
            }
        candidates.sort(key=lambda r: len(r['substances_canonical']))
        rep = candidates[0]
        return {
            'original_name': f"{canonical} (вещество, представитель: {rep['original_name']})",
            'substances_raw': [canonical],
            'substances_canonical': [canonical],
            'uses': rep['uses'],
            'side_effects': rep['side_effects'],
            'manufacturer': rep['manufacturer'],
        }

    # ============================================================
    # Главный метод: мульти-препаратный анализ
    # ============================================================
    def analyze_multi(self,
                      drug_queries: list[str],
                      user_prompt: str = '',
                      image_path: str | None = None,
                      ocr_languages: tuple[str, ...] = ('en', 'ru'),
                      use_gpu: bool = False,
                      model: str = DEFAULT_MODEL) -> AssistantResult:
        """
        Принимает список названий препаратов + опциональное фото + текст вопроса.
        Каждое название проходит resolve_one и становится DrugEntry.
        Если приложен image_path - первый препарат добавляется из OCR.
        Если в user_prompt упомянуты препараты - они также добавляются как DrugEntry.
        """
        result = AssistantResult()
        drugs: list[DrugEntry] = []
        seen_names = set()  # чтобы не дублировать одно и то же

        def add_unique(entry: DrugEntry):
            key = (entry.name or entry.query).lower()
            if key and key not in seen_names:
                drugs.append(entry)
                seen_names.add(key)

        # 1) Препарат с фото (если есть)
        if image_path:
            ocr_lines, ocr_error = self._safe_ocr(image_path, ocr_languages, use_gpu)
            result.ocr_lines = ocr_lines
            result.ocr_error = ocr_error

            if ocr_lines and not ocr_error:
                rec = find_medicine_name(ocr_lines, self.medicine_index)
                if rec:
                    entry = DrugEntry(
                        source='photo',
                        # query - именно тот фрагмент OCR, что сматчился, а не весь текст
                        query=rec.get('matched_query', rec['original_name']),
                        found=True,
                        name=rec['original_name'],
                        substances_raw=rec['substances_raw'],
                        substances_canonical=rec['substances_canonical'],
                        uses=rec['uses'],
                        side_effects=rec['side_effects'],
                        manufacturer=rec['manufacturer'],
                    )
                    add_unique(entry)
                else:
                    # Fallback: ищем по веществам в распознанных строках
                    found_via_substance = False
                    for line in ocr_lines:
                        for token in re.findall(r'[\w\-]{5,}', line, flags=re.UNICODE):
                            try:
                                from synonyms import translate_ru_to_en
                                en_token, _ = translate_ru_to_en(token)
                                token = en_token if en_token != token else token
                            except ImportError:
                                pass
                            sub_rec = self._find_by_substance(token)
                            if sub_rec:
                                entry = DrugEntry(
                                    source='photo', query=token, found=True,
                                    name=sub_rec['original_name'],
                                    substances_raw=sub_rec['substances_raw'],
                                    substances_canonical=sub_rec['substances_canonical'],
                                    uses=sub_rec['uses'],
                                    side_effects=sub_rec['side_effects'],
                                    manufacturer=sub_rec['manufacturer'],
                                )
                                add_unique(entry)
                                found_via_substance = True
                                break
                        if found_via_substance:
                            break
                    if not found_via_substance:
                        # Не нашли. Берём самый короткий вменяемый фрагмент,
                        # а не весь длинный текст.
                        best_guess = self._pick_best_name_guess(ocr_lines)
                        add_unique(DrugEntry(source='photo', query=best_guess, found=False))

        # 2) Препараты из явного списка (поле «Список препаратов»)
        for query in drug_queries:
            if query and query.strip():
                entry = self.resolve_one(query.strip(), source='manual')
                add_unique(entry)

        # 3) Препараты из текста вопроса (если упомянуты ещё какие-то)
        prompt_drugs = self._extract_drugs_from_prompt(user_prompt)
        for canon_substance in prompt_drugs:
            if canon_substance.lower() in seen_names:
                continue
            sub_rec = self._find_by_substance(canon_substance)
            if sub_rec:
                entry = DrugEntry(
                    source='prompt', query=canon_substance, found=True,
                    name=sub_rec['original_name'],
                    substances_raw=sub_rec['substances_raw'],
                    substances_canonical=sub_rec['substances_canonical'],
                    uses=sub_rec['uses'],
                    side_effects=sub_rec['side_effects'],
                    manufacturer=sub_rec['manufacturer'],
                )
                add_unique(entry)

        # 4) Сборка отчёта (со всеми кросс-проверками)
        result.report = build_report(drugs, user_prompt, self.interaction_index)

        # 4.1) Если препарат только один (либо все остальные не найдены) -
        # показываем N образцовых взаимодействий из базы, чтобы пользователь
        # видел, с чем препарат конфликтует, даже без явного второго препарата.
        found_drugs = [d for d in drugs if d.found]
        if len(found_drugs) == 1 and len(result.report.conflicts) == 0:
            from search_engine import find_general_interactions
            single = found_drugs[0]
            general = find_general_interactions(
                single.substances_canonical,
                self.interaction_index,
                max_results=4,
                random_seed=hash(single.name) & 0xFFFFFFFF,  # стабильный для одного препарата
            )
            # Помечаем, что это «общие» примеры, не явные пары
            for g in general:
                g['source_a'] = single.name
                g['source_b'] = '(пример из базы)'
                g['within_same_drug'] = False
                g['is_general_example'] = True
            result.report.conflicts = general
            if general:
                result.report.notes.append(
                    f"ℹ Показаны {len(general)} примера из базы — препараты, "
                    f"с которыми {single.name} имеет известные взаимодействия."
                )

        result.report_text = format_report_for_llm(result.report)

        # 5) Если ни одного препарата не найдено - не зовём LLM
        any_found = any(d.found for d in drugs)
        if not any_found:
            language = detect_language(user_prompt)
            attempted_list = [d.query for d in drugs if d.query]
            attempted = ', '.join(f'"{q}"' for q in attempted_list) if attempted_list else ''

            if language == 'ru':
                msg = (
                    f"🤔 К сожалению, я не нашёл ни один из указанных препаратов "
                    f"в моей базе{': ' + attempted if attempted else ''}.\n\n"
                    "Возможные причины:\n"
                    "• Препарат отсутствует в справочнике (~11 800 препаратов, индийский рынок).\n"
                    "• Текст с упаковки распознан некорректно — попробуйте ввести название вручную.\n"
                    "• Использовано торговое название, которого нет в базе — попробуйте "
                    "международное непатентованное название (МНН).\n\n"
                    "💡 Уточните название препарата или укажите действующее вещество."
                )
            else:
                msg = (
                    f"🤔 Sorry, I couldn't find any of the specified drugs in my database"
                    f"{': ' + attempted if attempted else ''}.\n\n"
                    "Possible reasons:\n"
                    "• The drug is not in the reference (~11,800 drugs, Indian market).\n"
                    "• OCR misread the package — try entering the name manually.\n"
                    "• You used a trade name not in the database — try the INN.\n\n"
                    "💡 Please clarify the drug name or specify the active ingredient."
                )
            result.llm = LLMResponse(text=msg, used_fallback=False)
            return result

        # 6) LLM. Если выбрана дообученная модель ("finetuned") — локальный
        # бэкенд (Qwen+LoRA через transformers). Иначе — Ollama как обычно.
        if model == "finetuned":
            from local_llm import generate_advice_local, is_available
            if is_available():
                result.llm = generate_advice_local(
                    result.report_text,
                    user_prompt_text=user_prompt,
                )
            else:
                result.llm = LLMResponse(
                    text="Дообученная модель недоступна: проверьте, что папка "
                         "адаптера на месте и установлены torch/transformers/peft.",
                    used_fallback=True,
                    error="finetuned backend unavailable",
                )
        else:
            result.llm = generate_advice(
                result.report_text,
                user_prompt_text=user_prompt,
                model=model,
            )
        return result

    def _extract_drugs_from_prompt(self, prompt: str) -> list[str]:
        """Извлекает упоминания препаратов из произвольного текста (как раньше делал extract_additional_drugs)."""
        if not prompt or not prompt.strip():
            return []

        try:
            from synonyms import translate_ru_to_en
            translated_prompt, _ = translate_ru_to_en(prompt)
        except ImportError:
            translated_prompt = prompt

        tokens = re.findall(r'[\w\-]{3,}', translated_prompt, flags=re.UNICODE)
        tokens += re.findall(r'[\w\-]{3,}', prompt, flags=re.UNICODE)

        # Стоп-слова
        STOP = {'tablet', 'capsule', 'syrup', 'mg', 'ml', 'таблетки', 'капсулы', 'мг', 'мл',
                'and', 'with', 'the', 'and', 'are', 'can', 'safe', 'together',
                'принимаю', 'вместе', 'это', 'эти', 'можно', 'можно ли', 'безопасно'}
        tokens = [t for t in tokens if len(t) >= 3 and t.lower() not in STOP and not t.isdigit()]

        found_canonical = []
        seen = set()
        for tok in tokens:
            tok_lower = tok.lower()
            canon = None
            if tok_lower in self.alias_map:
                canon = self.alias_map[tok_lower]
            else:
                # Fuzzy через _find_by_substance (он строгий)
                rec = self._find_by_substance(tok_lower)
                if rec:
                    # Берём только канонические вещества
                    for s in rec['substances_canonical']:
                        if s and s.lower() not in seen:
                            found_canonical.append(s)
                            seen.add(s.lower())
            if canon and canon.lower() not in seen:
                found_canonical.append(canon)
                seen.add(canon.lower())
        return found_canonical[:10]

    # ============================================================
    # Совместимость со старым API (один препарат)
    # ============================================================
    def analyze(self,
                image_path: str | None,
                user_prompt: str,
                ocr_languages: tuple[str, ...] = ('en', 'ru'),
                use_gpu: bool = False,
                model: str = DEFAULT_MODEL,
                manual_medicine_name: str | None = None) -> AssistantResult:
        """Старый API - оборачивает analyze_multi."""
        drug_queries = []
        if manual_medicine_name and manual_medicine_name.strip():
            drug_queries.append(manual_medicine_name.strip())
        return self.analyze_multi(
            drug_queries=drug_queries,
            user_prompt=user_prompt or '',
            image_path=image_path,
            ocr_languages=ocr_languages,
            use_gpu=use_gpu,
            model=model,
        )


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent
    asst = MedAssistant(cache_dir=base / 'cache')

    # Демо: 3 препарата сразу + один комбинированный
    print("=== Тест мульти-поиска: Augmentin (комбинированный) + Aspirin + Probenecid ===\n")
    res = asst.analyze_multi(
        drug_queries=['Augmentin 625', 'Aspirin', 'Probenecid'],
        user_prompt='Можно ли принимать вместе?',
    )
    print("=== ТЕХНИЧЕСКИЙ ОТЧЁТ ===")
    print(res.report_text)
