"""
Словари синонимов и нормализации названий действующих веществ.

INN_SYNONYMS - связывает разные названия одного и того же вещества
(как правило British vs USAN, или старые vs новые названия по WHO INN).
RU_TO_EN - русскоязычные торговые/МНН названия -> английские эквиваленты,
которые встречаются в Medicine_Details.csv или db_drug_interactions.csv.
"""

# Канонические синонимы: одно вещество -> разные написания.
# При сопоставлении составов с базой interactions проверяем все эти варианты.
INN_SYNONYMS = {
    # Боль/жар
    'paracetamol': ['acetaminophen'],
    'acetaminophen': ['paracetamol'],
    'metamizole': ['dipyrone', 'analgin', 'novalgin'],
    'dipyrone': ['metamizole', 'analgin'],

    # Антибиотики
    'amoxycillin': ['amoxicillin'],
    'amoxicillin': ['amoxycillin'],
    'sulphamethoxazole': ['sulfamethoxazole'],
    'sulfamethoxazole': ['sulphamethoxazole'],
    'cephalexin': ['cefalexin'],
    'cefalexin': ['cephalexin'],
    'cefuroxime': ['cefuroxime axetil'],
    'phenoxymethylpenicillin': ['penicillin v'],

    # Сердечные/гипотензивные
    'frusemide': ['furosemide'],
    'furosemide': ['frusemide'],
    'amlodipine': ['amlodipine besylate', 'amlodipine besilate'],
    'lisinopril': ['lisinopril dihydrate'],

    # Антигистамины
    'loratadine': ['claritin'],
    'cetirizine': ['cetirizine hydrochloride'],

    # ЖКТ
    'omeprazole': ['omeprazole magnesium'],
    'pantoprazole': ['pantoprazole sodium'],
    'ranitidine': ['ranitidine hydrochloride'],

    # Прочее
    'levothyroxine': ['l-thyroxine', 'thyroxine'],
    'salbutamol': ['albuterol'],
    'albuterol': ['salbutamol'],
    'adrenaline': ['epinephrine'],
    'epinephrine': ['adrenaline'],
    'noradrenaline': ['norepinephrine'],
    'norepinephrine': ['noradrenaline'],
    'lignocaine': ['lidocaine'],
    'lidocaine': ['lignocaine'],

    # Аспирин - в interactions он под химическим названием
    'aspirin': ['acetylsalicylic acid'],
    'acetylsalicylic acid': ['aspirin'],

    # Витамин B12 - methyl- и cyano- формы взаимозаменяемы для интеракций
    'methylcobalamin': ['cyanocobalamin'],
    'cyanocobalamin': ['methylcobalamin'],
    'hydroxocobalamin': ['cyanocobalamin'],

    # Противовирусные/прочие частые
    'aciclovir': ['acyclovir'],
    'acyclovir': ['aciclovir'],
    'azithromycin': ['azithromycin dihydrate'],
    'clarithromycin': ['biaxin'],
}


# Русские -> английские. Включает торговые названия, которые часто
# спрашивают на постсоветском пространстве.
RU_TO_EN = {
    # Анальгетики/жаропонижающие
    'аспирин': 'aspirin',
    'ацетилсалициловая кислота': 'aspirin',
    'парацетамол': 'paracetamol',
    'панадол': 'paracetamol',
    'эффералган': 'paracetamol',
    'анальгин': 'metamizole',
    'метамизол': 'metamizole',
    'ибупрофен': 'ibuprofen',
    'нурофен': 'ibuprofen',
    'мига': 'ibuprofen',
    'кеторолак': 'ketorolac',
    'кетанов': 'ketorolac',
    'кеторол': 'ketorolac',
    'диклофенак': 'diclofenac',
    'вольтарен': 'diclofenac',
    'нимесулид': 'nimesulide',
    'найз': 'nimesulide',
    'мелоксикам': 'meloxicam',

    # Антибиотики
    'амоксициллин': 'amoxicillin',
    'амоксиклав': 'amoxicillin',
    'аугментин': 'amoxicillin',
    'азитромицин': 'azithromycin',
    'сумамед': 'azithromycin',
    'азитрал': 'azithromycin',
    'кларитромицин': 'clarithromycin',
    'клацид': 'clarithromycin',
    'цефтриаксон': 'ceftriaxone',
    'ципрофлоксацин': 'ciprofloxacin',
    'ципролет': 'ciprofloxacin',
    'левофлоксацин': 'levofloxacin',
    'доксициклин': 'doxycycline',
    'метронидазол': 'metronidazole',
    'трихопол': 'metronidazole',

    # Кардиологические/гипотензивные
    'эналаприл': 'enalapril',
    'энап': 'enalapril',
    'лизиноприл': 'lisinopril',
    'периндоприл': 'perindopril',
    'престариум': 'perindopril',
    'амлодипин': 'amlodipine',
    'норваск': 'amlodipine',
    'бисопролол': 'bisoprolol',
    'конкор': 'bisoprolol',
    'метопролол': 'metoprolol',
    'эгилок': 'metoprolol',
    'атенолол': 'atenolol',
    'лозартан': 'losartan',
    'лориста': 'losartan',
    'валсартан': 'valsartan',
    'фуросемид': 'furosemide',
    'лазикс': 'furosemide',
    'индапамид': 'indapamide',
    'арифон': 'indapamide',
    'нитроглицерин': 'nitroglycerin',
    'аторвастатин': 'atorvastatin',
    'липримар': 'atorvastatin',
    'розувастатин': 'rosuvastatin',
    'крестор': 'rosuvastatin',

    # Антикоагулянты/антиагреганты
    'варфарин': 'warfarin',
    'клопидогрел': 'clopidogrel',
    'плавикс': 'clopidogrel',
    'гепарин': 'heparin',

    # Сахарный диабет
    'метформин': 'metformin',
    'сиофор': 'metformin',
    'глюкофаж': 'metformin',
    'инсулин': 'insulin',
    'гликлазид': 'gliclazide',

    # ЖКТ
    'омепразол': 'omeprazole',
    'омез': 'omeprazole',
    'пантопразол': 'pantoprazole',
    'нольпаза': 'pantoprazole',
    'эзомепразол': 'esomeprazole',
    'нексиум': 'esomeprazole',
    'ранитидин': 'ranitidine',
    'фамотидин': 'famotidine',
    'квамател': 'famotidine',
    'лоперамид': 'loperamide',
    'имодиум': 'loperamide',
    'дротаверин': 'drotaverine',
    'но-шпа': 'drotaverine',
    'мебеверин': 'mebeverine',
    'дюспаталин': 'mebeverine',

    # Антигистамины
    'лоратадин': 'loratadine',
    'кларитин': 'loratadine',
    'цетиризин': 'cetirizine',
    'зиртек': 'cetirizine',
    'зодак': 'cetirizine',
    'дезлоратадин': 'desloratadine',
    'эриус': 'desloratadine',
    'супрастин': 'chloropyramine',
    'тавегил': 'clemastine',

    # Дыхательная система
    'сальбутамол': 'salbutamol',
    'вентолин': 'salbutamol',
    'беродуал': 'ipratropium',
    'будесонид': 'budesonide',
    'пульмикорт': 'budesonide',
    'амброксол': 'ambroxol',
    'лазолван': 'ambroxol',
    'ацетилцистеин': 'acetylcysteine',
    'ацц': 'acetylcysteine',
    'бромгексин': 'bromhexine',

    # Психотропные/успокоительные
    'диазепам': 'diazepam',
    'реланиум': 'diazepam',
    'феназепам': 'phenazepam',
    'флуоксетин': 'fluoxetine',
    'прозак': 'fluoxetine',
    'сертралин': 'sertraline',
    'золофт': 'sertraline',
    'амитриптилин': 'amitriptyline',

    # Гормональные / щитовидка
    'левотироксин': 'levothyroxine',
    'эутирокс': 'levothyroxine',
    'l-тироксин': 'levothyroxine',
    'преднизолон': 'prednisolone',
    'дексаметазон': 'dexamethasone',
    'гидрокортизон': 'hydrocortisone',

    # Противовирусные
    'ацикловир': 'acyclovir',
    'зовиракс': 'acyclovir',
    'осельтамивир': 'oseltamivir',
    'тамифлю': 'oseltamivir',

    # Мочевыводящие/прочее
    'фурагин': 'furazidin',
    'нолицин': 'norfloxacin',
    'канефрон': 'canephron',

    # Витамины и минералы (как обозначены в interactions)
    'витамин d': 'cholecalciferol',
    'витамин с': 'ascorbic acid',
    'аскорбиновая кислота': 'ascorbic acid',
    'тиамин': 'thiamine',
    'витамин в1': 'thiamine',
    'пиридоксин': 'pyridoxine',
    'витамин в6': 'pyridoxine',
    'цианокобаламин': 'cyanocobalamin',
    'витамин в12': 'cyanocobalamin',
    'фолиевая кислота': 'folic acid',
}


def normalize_substance_name(name: str) -> str:
    """Базовая нормализация: lower, удаление солевых форм/USP/BP, орфографические замены."""
    if not name:
        return ''
    s = name.lower().strip()

    # Удаляем стандартные «хвосты» вроде солей и квалификаторов
    suffixes = [
        ' hydrochloride', ' hcl', ' sulfate', ' sulphate',
        ' sodium', ' potassium', ' calcium', ' magnesium',
        ' citrate', ' phosphate', ' tartrate', ' maleate',
        ' besylate', ' besilate', ' fumarate', ' succinate',
        ' dihydrate', ' monohydrate', ' trihydrate',
        ' (usp)', ' (bp)', ' usp', ' bp',
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)].strip()
                changed = True

    # Орфографические нормализации British/American
    repls = [
        ('sulph', 'sulf'),
        ('oestro', 'estro'),
        ('haema', 'hema'),
        ('aluminium', 'aluminum'),
    ]
    for old, new in repls:
        s = s.replace(old, new)

    # Уменьшаем кратные пробелы
    import re as _re
    s = _re.sub(r'\s+', ' ', s).strip()
    return s


def expand_with_synonyms(name: str) -> list[str]:
    """Возвращает варианты имени для перебора при сопоставлении."""
    norm = normalize_substance_name(name)
    variants = {name.lower(), norm}
    if norm in INN_SYNONYMS:
        for syn in INN_SYNONYMS[norm]:
            variants.add(syn)
            variants.add(normalize_substance_name(syn))
    return [v for v in variants if v]


def translate_ru_to_en(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Заменяет упоминания русских препаратов на английские.
    Возвращает (новый_текст, список_замен).
    Учитывает русские падежные окончания: 'Аспирином', 'Ибупрофена', 'Парацетамолу' и т.п.
    """
    if not text:
        return text, []
    import re as _re
    new_text = text
    replacements = []

    # Список окончаний (от длинного к короткому, чтобы 'ами' матчилось раньше 'а')
    russian_endings = ['ами', 'ями', 'ом', 'ем', 'ой', 'ей', 'ах', 'ях',
                       'ы', 'и', 'у', 'ю', 'е', 'а', 'я']

    # Сортируем по длине (длинные сначала, чтобы 'но-шпа' матчилось раньше 'но')
    for ru in sorted(RU_TO_EN.keys(), key=len, reverse=True):
        en = RU_TO_EN[ru]
        # Строим паттерн: основа + опциональное окончание
        endings_alt = '|'.join(russian_endings)
        pattern = _re.compile(
            r'(?<![\w\u0400-\u04FF])'
            + _re.escape(ru)
            + r'(?:' + endings_alt + r')?'
            + r'(?![\w\u0400-\u04FF])',
            flags=_re.IGNORECASE
        )
        match = pattern.search(new_text)
        if match:
            new_text = pattern.sub(en, new_text)
            replacements.append((ru, en))
    return new_text, replacements
