"""
Этап 1: Подготовка данных (Data Engineering)
- Очистка Composition от дозировок (регулярки)
- Создание индекса: торговое название -> список действующих веществ
- Fuzzy-связка веществ из Medicine_Details с веществами из db_drug_interactions
"""
import re
import pickle
from pathlib import Path
import pandas as pd

# Предпочитаем rapidfuzz (быстрее), затем thefuzz, затем встроенный difflib
try:
    from rapidfuzz import process, fuzz
    _FUZZ_BACKEND = 'rapidfuzz'
except ImportError:
    try:
        from thefuzz import process, fuzz
        _FUZZ_BACKEND = 'thefuzz'
    except ImportError:
        import difflib
        _FUZZ_BACKEND = 'difflib'

        class _DifflibScorer:
            @staticmethod
            def WRatio(a, b):
                return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)

            @staticmethod
            def token_set_ratio(a, b):
                """Эмуляция token_set_ratio: пересечение токенов + симметричный остаток."""
                ta, tb = set(a.lower().split()), set(b.lower().split())
                if not ta or not tb:
                    return 0
                inter = ta & tb
                if not inter:
                    return _DifflibScorer.WRatio(a, b)
                # Если есть пересечение, увеличиваем оценку
                base = _DifflibScorer.WRatio(' '.join(sorted(inter)), b)
                full = _DifflibScorer.WRatio(a, b)
                return max(base, full)

            @staticmethod
            def partial_ratio(a, b):
                """Эмуляция partial_ratio: ищет подстроку короткой строки в длинной."""
                if not a or not b:
                    return 0
                shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
                if shorter in longer:
                    return 100
                # SequenceMatcher с фокусом на подстроке
                m = difflib.SequenceMatcher(None, shorter, longer)
                blocks = m.get_matching_blocks()
                best = 0
                for block in blocks:
                    if block.size == 0:
                        continue
                    start = max(0, block.b - block.a)
                    end = start + len(shorter)
                    sub = longer[start:end]
                    score = int(difflib.SequenceMatcher(None, shorter, sub).ratio() * 100)
                    best = max(best, score)
                return best

        class _DifflibProcess:
            @staticmethod
            def extractOne(query, choices, scorer=None, score_cutoff=0):
                if scorer is None:
                    scorer = _DifflibScorer.WRatio
                best = None
                best_score = -1
                for c in choices:
                    s = scorer(query, c)
                    if s > best_score:
                        best_score = s
                        best = c
                if best is None or best_score < score_cutoff:
                    return None
                return (best, best_score, 0)

        process = _DifflibProcess()
        fuzz = _DifflibScorer()


# Регулярка для удаления дозировок: (500mg), (1mg/5ml), (5 mg), (10%), и т.п.
DOSAGE_PATTERN = re.compile(
    r'\s*\([^)]*?(?:mg|ml|mcg|g|iu|%|units?|mmol|mEq)[^)]*?\)',
    flags=re.IGNORECASE
)
# Запасная регулярка - убирает любые оставшиеся скобки с числами
PARENS_WITH_DIGITS = re.compile(r'\s*\([^)]*\d[^)]*\)')


def clean_composition(raw: str) -> list[str]:
    """
    'Amoxycillin  (500mg) +  Clavulanic Acid (125mg)' ->
    ['Amoxycillin', 'Clavulanic Acid']
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    text = DOSAGE_PATTERN.sub('', raw)
    text = PARENS_WITH_DIGITS.sub('', text)

    # Разделяем по '+' (комбинированные препараты)
    parts = [p.strip() for p in text.split('+')]
    # Финальная чистка: убираем лишние пробелы, пустые элементы
    parts = [re.sub(r'\s+', ' ', p).strip() for p in parts if p.strip()]
    return parts


def build_drug_alias_map(medicine_df: pd.DataFrame,
                        interaction_drugs: set[str],
                        score_cutoff: int = 82) -> dict[str, str]:
    """
    Сопоставляет каждое уникальное вещество из Medicine_Details
    с ближайшим названием в db_drug_interactions через fuzzy matching.

    Стратегия:
      1. Точное совпадение нормализованного имени.
      2. Проверка всех INN-синонимов (paracetamol↔acetaminophen и т.п.).
      3. Fuzzy-поиск через token_set_ratio (устойчив к перестановкам слов
         и солевым хвостам), запасной WRatio.

    Возвращает: {composition_name_lower: canonical_interaction_name}.
    """
    from synonyms import normalize_substance_name, expand_with_synonyms

    # Собираем все уникальные вещества из состава
    all_substances = set()
    for comp_str in medicine_df['Composition'].dropna():
        for sub in clean_composition(comp_str):
            if sub:
                all_substances.add(sub)

    print(f"  Уникальных веществ в Medicine_Details: {len(all_substances)}")
    print(f"  Уникальных веществ в interactions: {len(interaction_drugs)}")

    # Готовим индекс канонических названий: и оригинал, и нормализованная форма
    canon_lower_to_original = {}        # 'amoxicillin' -> 'Amoxicillin'
    canon_normalized_to_original = {}   # после удаления солевых хвостов
    for d in interaction_drugs:
        d_lower = d.lower().strip()
        canon_lower_to_original[d_lower] = d
        norm = normalize_substance_name(d)
        if norm and norm != d_lower:
            canon_normalized_to_original.setdefault(norm, d)

    canon_lower_list = list(canon_lower_to_original.keys())
    canon_norm_list = list(canon_normalized_to_original.keys())

    alias_map = {}
    matched_exact = matched_synonym = matched_fuzzy = unmatched = 0

    for sub in all_substances:
        sub_lower = sub.lower()

        # 1) Точное совпадение
        if sub_lower in canon_lower_to_original:
            alias_map[sub_lower] = canon_lower_to_original[sub_lower]
            matched_exact += 1
            continue

        # 2) Проверяем все варианты (нормализация + синонимы)
        found = False
        for variant in expand_with_synonyms(sub):
            if variant in canon_lower_to_original:
                alias_map[sub_lower] = canon_lower_to_original[variant]
                matched_synonym += 1
                found = True
                break
            if variant in canon_normalized_to_original:
                alias_map[sub_lower] = canon_normalized_to_original[variant]
                matched_synonym += 1
                found = True
                break
        if found:
            continue

        # 3) Fuzzy match - сначала по нормализованным названиям
        sub_norm = normalize_substance_name(sub)
        try:
            # token_set_ratio - устойчив к разному порядку слов и лишним токенам
            scorer = fuzz.token_set_ratio if hasattr(fuzz, 'token_set_ratio') else fuzz.WRatio
            result = process.extractOne(
                sub_norm or sub_lower, canon_norm_list,
                scorer=scorer, score_cutoff=score_cutoff
            )
            if result:
                best_norm = result[0]
                alias_map[sub_lower] = canon_normalized_to_original[best_norm]
                matched_fuzzy += 1
                continue

            # Fallback - по сырым именам
            result = process.extractOne(
                sub_lower, canon_lower_list,
                scorer=scorer, score_cutoff=score_cutoff
            )
            if result:
                alias_map[sub_lower] = canon_lower_to_original[result[0]]
                matched_fuzzy += 1
                continue
        except Exception:
            pass

        unmatched += 1

    matched = matched_exact + matched_synonym + matched_fuzzy
    print(f"  Сопоставлено: {matched} ({matched_exact} точно + "
          f"{matched_synonym} по синонимам + {matched_fuzzy} fuzzy) | "
          f"без пары: {unmatched}")
    return alias_map


def build_medicine_index(medicine_df: pd.DataFrame,
                         alias_map: dict[str, str]) -> dict[str, dict]:
    """
    Индекс: торговое название (lower) -> {
        'original_name': str,
        'substances_raw': [...],     # как в Composition
        'substances_canonical': [...],  # после fuzzy matching - для поиска по interactions
        'uses': str,
        'side_effects': str,
        'manufacturer': str,
    }
    """
    index = {}
    for _, row in medicine_df.iterrows():
        name = str(row['Medicine Name']).strip()
        substances_raw = clean_composition(row.get('Composition', ''))
        substances_canon = []
        for s in substances_raw:
            canon = alias_map.get(s.lower())
            if canon:
                substances_canon.append(canon)

        index[name.lower()] = {
            'original_name': name,
            'substances_raw': substances_raw,
            'substances_canonical': substances_canon,
            'uses': str(row.get('Uses', '')).strip(),
            'side_effects': str(row.get('Side_effects', '')).strip(),
            'manufacturer': str(row.get('Manufacturer', '')).strip(),
        }
    return index


def build_interaction_index(interactions_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """
    Индекс пар (drug_a_lower, drug_b_lower) -> описание взаимодействия.
    Симметричный: храним в обе стороны для быстрого поиска.
    """
    idx = {}
    for _, row in interactions_df.iterrows():
        d1 = str(row['Drug 1']).strip()
        d2 = str(row['Drug 2']).strip()
        desc = str(row['Interaction Description']).strip()
        key1 = (d1.lower(), d2.lower())
        key2 = (d2.lower(), d1.lower())
        idx[key1] = desc
        if key2 not in idx:
            idx[key2] = desc
    return idx


def prepare_all(medicine_csv: str, interactions_csv: str, cache_dir: str):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Чтение Medicine_Details.csv...")
    med_df = pd.read_csv(medicine_csv)
    print(f"      Загружено {len(med_df)} препаратов")

    print("[2/4] Чтение db_drug_interactions.csv...")
    inter_df = pd.read_csv(interactions_csv)
    print(f"      Загружено {len(inter_df)} пар взаимодействий")

    interaction_drugs = set()
    interaction_drugs.update(inter_df['Drug 1'].dropna().astype(str).str.strip())
    interaction_drugs.update(inter_df['Drug 2'].dropna().astype(str).str.strip())

    print("[3/4] Fuzzy-сопоставление веществ (это самый долгий шаг)...")
    alias_map = build_drug_alias_map(med_df, interaction_drugs)

    print("[4/4] Построение индексов...")
    medicine_index = build_medicine_index(med_df, alias_map)
    interaction_index = build_interaction_index(inter_df)

    # Сохраняем в кэш
    with open(cache_dir / 'medicine_index.pkl', 'wb') as f:
        pickle.dump(medicine_index, f)
    with open(cache_dir / 'interaction_index.pkl', 'wb') as f:
        pickle.dump(interaction_index, f)
    with open(cache_dir / 'alias_map.pkl', 'wb') as f:
        pickle.dump(alias_map, f)

    print(f"\n✓ Готово. Кэш сохранён в {cache_dir}")
    print(f"  medicine_index: {len(medicine_index)} записей")
    print(f"  interaction_index: {len(interaction_index)} ключей")
    return medicine_index, interaction_index, alias_map


def load_cache(cache_dir: str):
    cache_dir = Path(cache_dir)
    with open(cache_dir / 'medicine_index.pkl', 'rb') as f:
        medicine_index = pickle.load(f)
    with open(cache_dir / 'interaction_index.pkl', 'rb') as f:
        interaction_index = pickle.load(f)
    with open(cache_dir / 'alias_map.pkl', 'rb') as f:
        alias_map = pickle.load(f)
    return medicine_index, interaction_index, alias_map


if __name__ == '__main__':
    import sys
    base = Path(__file__).resolve().parent.parent
    prepare_all(
        medicine_csv=str(base / 'data' / 'Medicine_Details.csv'),
        interactions_csv=str(base / 'data' / 'db_drug_interactions.csv'),
        cache_dir=str(base / 'cache'),
    )
