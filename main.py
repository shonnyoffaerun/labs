"""
Модуль: main.py
Автор: Лонщаков Александр, МТ-201
Назначение: Главная точка входа в проект (CLI Pipeline).
            Интегрирует все компоненты команды (data_loader, preprocessor, search_engine, evaluator),
            запускает аудит корпуса, базовый поиск, абляционное исследование TF-IDF,
            детальный разбор ошибок и экспорт финальных CSV-файлов.
"""

import os
import sys

# Добавляем путь к текущей директории для гарантированного импорта модулей
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from search_engine import WordOverlapSearcher
from evaluator import MetricsEvaluator


def main():
    print("=" * 70)
    print("🏛️ ЛАБОРАТОРНАЯ РАБОТА № 2: УМНЫЙ ПОИСК ПО ДОКУМЕНТАМ")
    print("   Тематика: Вариант 4 — Музейный каталог искусства")
    print("   Авторы проекта: Кравченко Д., Тюлин К., Примакова А., Шагиев Д., Лонщаков А.")
    print("=" * 70)

    # 1. Загрузка и исследование данных (Модуль Кравченко Дмитрия)
    loader = DataLoader()
    docs_df, queries_df, rel_df = loader.load_data()
    loader.inspect_corpus()
    doc_texts, doc_ids = loader.get_full_documents_texts()

    # 2. Инициализация лингвистического препроцессора (Модуль Примаковой Анны)
    print("⏳ Инициализация токенизатора, словаря стоп-слов и pymorphy3...")
    preprocessor = TextPreprocessor()

    # 3. Базовый поиск по совпадению слов — Baseline (Модуль Тюлина Константина)
    print("\n" + "=" * 65)
    print("🔎 БАЗОВЫЙ ПОИСК: WORD OVERLAP BASELINE (Критерий 3)")
    print("=" * 65)
    overlap_searcher = WordOverlapSearcher(doc_texts, doc_ids)
    baseline_results = overlap_searcher.batch_search(queries_df, top_k=5)

    # Оценка базового поиска (Модуль Шагиева Даниила)
    evaluator = MetricsEvaluator(rel_df)
    base_hit, base_mrr = evaluator.evaluate_results(baseline_results, top_k=5)
    print(f"Базовый поиск (пересечение слов): Hit@5 = {base_hit:.4f} | MRR@5 = {base_mrr:.4f}")

    # 4. Абляционное исследование TF-IDF: конфигурации A, B, C, D (Модуль Шагиева Даниила)
    metrics_df, results_by_config, best_config_name = evaluator.run_ablation_study(
        doc_texts=doc_texts,
        doc_ids=doc_ids,
        queries_df=queries_df,
        preprocessor=preprocessor
    )

    # 5. Детальный разбор ошибок поиска (Модуль Шагиева Даниила)
    evaluator.analyze_errors(
        results_by_config=results_by_config,
        docs_df=docs_df,
        queries_df=queries_df,
        baseline_results=baseline_results,
        top_k=5
    )

    # 6. Сохранение финальных отчетов (Критерий 7)
    # Сохраняем в папку запуска и на уровень выше (для обратной совместимости)
    best_results_df = results_by_config[best_config_name]
    output_dirs = [loader.data_dir]
    if current_dir not in output_dirs:
        output_dirs.append(current_dir)

    for out_dir in set(output_dirs):
        evaluator.export_results(metrics_df, best_results_df, output_dir=out_dir)

    print("\n" + "=" * 70)
    print("✅ ВСЕ РАСЧЕТЫ И ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("🌐 Для запуска интерактивного веб-сайта выполните в терминале:")
    print("   streamlit run app.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
