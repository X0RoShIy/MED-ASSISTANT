"""
Этап 3: Поисковое ядро (Logic Engine)
Без участия LLM - только детерминированный поиск, чтобы исключить галлюцинации.

Поддерживает мульти-поиск: можно передать N препаратов, и каждая пара (включая
пары их веществ внутри комбинированных препаратов) будет проверена против базы
взаимодействий.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class DrugEntry:
    """Один препарат в отчёте: торговое название + раскрытый состав."""
    source: str = ''                  # 'photo' | 'manual' | 'prompt'
    query: str = ''                   # как ввёл/распознался запрос пользователя
    found: bool = False               # удалось найти в базе?
    name: str = ''                    # название из базы (или название вещества)
    substances_raw: list[str] = field(default_factory=list)
    substances_canonical: list[str] = field(default_factory=list)
    uses: str = ''
    side_effects: str = ''
    manufacturer: str = ''


@dataclass
class TechnicalReport:
    """Структурированный отчёт для подачи в LLM (мульти-препарат)."""
    drugs: list[DrugEntry] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    user_prompt: str = ''
    notes: list[str] = field(default_factory=list)

    # === Совместимость со старым API GUI ===
    @property
    def main_drug_name(self) -> str:
        return self.drugs[0].name if self.drugs and self.drugs[0].found else ''

    @property
    def main_drug_substances(self) -> list[str]:
        return self.drugs[0].substances_raw if self.drugs and self.drugs[0].found else []

    @property
    def main_drug_substances_canonical(self) -> list[str]:
        return self.drugs[0].substances_canonical if self.drugs and self.drugs[0].found else []

    @property
    def main_drug_uses(self) -> str:
        return self.drugs[0].uses if self.drugs and self.drugs[0].found else ''

    @property
    def main_drug_side_effects(self) -> str:
        return self.drugs[0].side_effects if self.drugs and self.drugs[0].found else ''

    @property
    def main_drug_manufacturer(self) -> str:
        return self.drugs[0].manufacturer if self.drugs and self.drugs[0].found else ''

    @property
    def additional_drugs(self) -> list[str]:
        out = []
        for d in self.drugs[1:]:
            if d.found:
                out.extend(d.substances_canonical)
            elif d.query:
                out.append(d.query)
        return out

    def to_dict(self) -> dict:
        return {
            'drugs': [asdict(d) for d in self.drugs],
            'conflicts': self.conflicts,
            'user_prompt': self.user_prompt,
            'notes': self.notes,
        }


def find_general_interactions(substances_canonical: list[str],
                              interaction_index: dict,
                              max_results: int = 4,
                              random_seed: int | None = None) -> list[dict]:
    """
    Для одного препарата (или его веществ) находит N показательных взаимодействий
    из базы — вещества, с которыми у него есть конфликты.

    Используется когда пользователь ввёл только один препарат и хочет узнать,
    с чем его НЕЛЬЗЯ принимать. Возвращает максимум max_results элементов
    (приоритет: уникальные партнёры, не повторяющиеся).
    """
    import random

    if not substances_canonical:
        return []

    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    # Собираем все взаимодействия для каждого вещества
    found = []
    seen_partners = set()

    for sub in substances_canonical:
        sub_lower = sub.lower()
        # Все ключи interaction_index, в которых упоминается это вещество
        partners = []
        for (a, b), desc in interaction_index.items():
            if a == sub_lower:
                partners.append((b, desc))
            elif b == sub_lower:
                partners.append((a, desc))
        # Перемешиваем чтобы выбрать "случайных", а не первых попавшихся
        rng.shuffle(partners)
        for partner_lower, desc in partners:
            if partner_lower in seen_partners or partner_lower in [s.lower() for s in substances_canonical]:
                continue
            found.append({
                'drug_a': sub,
                'drug_b': partner_lower.title(),  # для отображения
                'description': desc,
            })
            seen_partners.add(partner_lower)
            if len(found) >= max_results:
                return found
    return found


def find_all_conflicts(drugs: list[DrugEntry], interaction_index: dict) -> list[dict]:
    """
    Полный мульти-поиск. Перебирает все пары веществ из всех препаратов:
      - внутри каждого комбинированного препарата (пары его собственных веществ)
      - между препаратами (все межкомбинации)

    Возвращает: [{drug_a, drug_b, description, source_a, source_b, within_same_drug}]
    """
    conflicts = []
    seen_pairs = set()

    flat: list[tuple[str, str]] = []  # (вещество, имя_препарата_к_которому_оно_относится)
    for d in drugs:
        if not d.found:
            continue
        for s in d.substances_canonical:
            if s:
                flat.append((s, d.name))

    n = len(flat)
    for i in range(n):
        sa, src_a = flat[i]
        for j in range(i + 1, n):
            sb, src_b = flat[j]
            if sa.lower() == sb.lower():
                continue
            canon_pair = tuple(sorted([sa.lower(), sb.lower()]))
            if canon_pair in seen_pairs:
                continue

            desc = (interaction_index.get((sa.lower(), sb.lower()))
                    or interaction_index.get((sb.lower(), sa.lower())))
            if desc:
                conflicts.append({
                    'drug_a': sa,
                    'drug_b': sb,
                    'description': desc,
                    'source_a': src_a,
                    'source_b': src_b,
                    'within_same_drug': src_a == src_b,
                })
                seen_pairs.add(canon_pair)

    return conflicts


def build_report(drugs: list[DrugEntry],
                 user_prompt: str,
                 interaction_index: dict) -> TechnicalReport:
    """Собирает мульти-препаратный технический отчёт."""
    report = TechnicalReport(user_prompt=user_prompt or '', drugs=list(drugs))

    not_found = [d.query for d in drugs if not d.found and d.query]
    if not_found:
        report.notes.append(
            f"Препараты, которых нет в справочнике (пропущены при анализе): {', '.join(not_found)}"
        )
    for d in drugs:
        if d.found and len(d.substances_raw) > len(d.substances_canonical):
            diff = len(d.substances_raw) - len(d.substances_canonical)
            report.notes.append(
                f"⚠ {d.name}: {diff} вещ-в не сопоставлено с базой взаимодействий"
            )

    report.conflicts = find_all_conflicts(drugs, interaction_index)
    return report


def format_report_for_llm(report: TechnicalReport) -> str:
    """Форматирует отчёт в текст для подачи в LLM."""
    lines = []
    found_drugs = [d for d in report.drugs if d.found]

    if not found_drugs:
        lines.append("Препараты: НИ ОДИН не найден в справочнике")
    else:
        lines.append(f"Анализируемые препараты ({len(found_drugs)}):")
        for i, d in enumerate(found_drugs, 1):
            lines.append(f"\n[{i}] {d.name}")
            if d.substances_raw:
                lines.append(f"    Действующие вещества: {', '.join(d.substances_raw)}")
            if d.uses:
                lines.append(f"    Применение: {d.uses}")
            if d.side_effects:
                lines.append(f"    Побочные эффекты: {d.side_effects}")
            if d.manufacturer:
                lines.append(f"    Производитель: {d.manufacturer}")

    not_found = [d for d in report.drugs if not d.found and d.query]
    if not_found:
        lines.append(f"\nПрепараты, не найденные в базе: {', '.join(d.query for d in not_found)}")

    if report.conflicts:
        # Группируем: общие примеры / внутри-препарата / между препаратами
        general = [c for c in report.conflicts if c.get('is_general_example')]
        within = [c for c in report.conflicts if c.get('within_same_drug') and not c.get('is_general_example')]
        between = [c for c in report.conflicts if not c.get('within_same_drug') and not c.get('is_general_example')]

        if general:
            lines.append(
                f"\nПримеры известных взаимодействий из базы "
                f"(препараты, с которыми указанное лекарство имеет конфликты, всего: {len(general)}):"
            )
            for c in general:
                lines.append(f"  • {c['drug_a']} ↔ {c['drug_b']}")
                lines.append(f"    {c['description']}")
            lines.append("    [Это примеры из базы - конкретно эти препараты пользователь не упоминал]")

        if within or between:
            lines.append(f"\nНайденные взаимодействия (всего: {len(within) + len(between)}):")
            if within:
                lines.append(f"\n  Внутри одного препарата (между его действующими веществами):")
                for c in within:
                    lines.append(f"    • [{c['source_a']}] {c['drug_a']} ↔ {c['drug_b']}")
                    lines.append(f"      {c['description']}")
            if between:
                lines.append(f"\n  Между разными препаратами:")
                for c in between:
                    lines.append(f"    • {c['drug_a']} (из {c['source_a']}) ↔ {c['drug_b']} (из {c['source_b']})")
                    lines.append(f"      {c['description']}")
    elif found_drugs:
        lines.append("\nВзаимодействий между указанными препаратами не найдено.")

    if report.user_prompt:
        lines.append(f"\nЗапрос пользователя: {report.user_prompt}")

    if report.notes:
        lines.append("\nТехнические заметки:")
        for n in report.notes:
            lines.append(f"  {n}")

    return '\n'.join(lines)


# ---- Обратная совместимость со старым кодом ----
def find_conflicts(substances_canonical, additional_drugs, interaction_index):
    """Устаревший API. Используйте find_all_conflicts."""
    drugs = [DrugEntry(name='main', found=True, substances_canonical=list(substances_canonical))]
    for ad in additional_drugs:
        drugs.append(DrugEntry(name=ad, found=True, substances_canonical=[ad]))
    return find_all_conflicts(drugs, interaction_index)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_preparation import load_cache

    base = Path(__file__).resolve().parent.parent
    medicine_index, interaction_index, _ = load_cache(str(base / 'cache'))

    # Демо мульти-поиска
    rec1 = medicine_index['augmentin 625 duo tablet']
    rec2 = medicine_index.get('aspisol 75 tablet') or list(medicine_index.values())[0]

    drugs = [
        DrugEntry(source='manual', query='Augmentin 625', found=True,
                  name=rec1['original_name'],
                  substances_raw=rec1['substances_raw'],
                  substances_canonical=rec1['substances_canonical'],
                  uses=rec1['uses'], side_effects=rec1['side_effects'],
                  manufacturer=rec1['manufacturer']),
        DrugEntry(source='manual', query='Aspirin', found=True,
                  name=rec2['original_name'],
                  substances_raw=rec2['substances_raw'],
                  substances_canonical=rec2['substances_canonical'],
                  uses=rec2['uses'], side_effects=rec2['side_effects'],
                  manufacturer=rec2['manufacturer']),
    ]
    rep = build_report(drugs, 'Можно ли вместе?', interaction_index)
    print(format_report_for_llm(rep))
