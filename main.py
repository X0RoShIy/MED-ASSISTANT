"""
Главный entry-point.
Запуск:
    python main.py            # запустить GUI (готовит кэш при первом запуске)
    python main.py --prepare  # только подготовить данные
    python main.py --cli      # CLI-режим без GUI
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

DATA_DIR = ROOT / 'data'
CACHE_DIR = ROOT / 'cache'


def ensure_cache():
    """Готовит индексы если кэша ещё нет."""
    needed = [CACHE_DIR / 'medicine_index.pkl',
              CACHE_DIR / 'interaction_index.pkl',
              CACHE_DIR / 'alias_map.pkl']
    if all(p.exists() for p in needed):
        return
    print("Кэш не найден - подготавливаю данные (один раз, ~1-2 минуты)...")
    from data_preparation import prepare_all
    prepare_all(
        medicine_csv=str(DATA_DIR / 'Medicine_Details.csv'),
        interactions_csv=str(DATA_DIR / 'db_drug_interactions.csv'),
        cache_dir=str(CACHE_DIR),
    )


def run_gui():
    ensure_cache()
    from gui import build_interface
    demo = build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


def run_cli():
    ensure_cache()
    from recommender import MedAssistant
    asst = MedAssistant(cache_dir=CACHE_DIR)

    print("\n=== Med Assistant CLI ===")
    img_path = input("Путь к фото (Enter, чтобы пропустить): ").strip() or None
    manual = input("Или название вручную (Enter, чтобы пропустить): ").strip() or None
    prompt = input("Ваш вопрос: ").strip()

    print("\n[анализ...]\n")
    result = asst.analyze(image_path=img_path, user_prompt=prompt, manual_medicine_name=manual)

    print("=== Технический отчёт ===")
    print(result.report_text)
    print("\n=== Ответ ассистента ===")
    print(result.llm.text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true', help='Только подготовить кэш')
    p.add_argument('--cli', action='store_true', help='CLI-режим')
    args = p.parse_args()

    if args.prepare:
        from data_preparation import prepare_all
        prepare_all(
            medicine_csv=str(DATA_DIR / 'Medicine_Details.csv'),
            interactions_csv=str(DATA_DIR / 'db_drug_interactions.csv'),
            cache_dir=str(CACHE_DIR),
        )
    elif args.cli:
        run_cli()
    else:
        run_gui()


if __name__ == '__main__':
    main()
