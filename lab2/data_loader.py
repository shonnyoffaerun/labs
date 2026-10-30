"""
Модуль: data_loader.py
Автор: Кравченко Дмитрий, МТ-201
Назначение: Загрузка исходных данных (documents.csv, queries.csv, dev_relevance.csv)
            в кодировке UTF-8 with BOM, первичный аудит целостности корпуса,
            проверка типов, пропусков, дубликатов и сборка полных текстов.
"""

import os
import pandas as pd


class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.docs_df = None
        self.queries_df = None
        self.rel_df = None
    def _resolve_data_dir(self, custom_dir):
        if custom_dir and os.path.exists(os.path.join(custom_dir, 'documents.csv')):
            return custom_dir

        if os.path.exists('documents.csv'):
            return '.'

        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if os.path.exists(os.path.join(parent_dir, 'documents.csv')):
            return parent_dir

        if os.path.exists(os.path.join(script_dir, 'documents.csv')):
            return script_dir

        return '.'

    def load_data(self):
        docs_path = os.path.join(self.data_dir, 'documents.csv')
        queries_path = os.path.join(self.data_dir, 'queries.csv')
        relevance_path = os.path.join(self.data_dir, 'dev_relevance.csv')

        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Файл {docs_path} не найден. Проверьте путь к данным.")

        self.docs_df = pd.read_csv(docs_path, encoding='utf-8-sig')
        self.queries_df = pd.read_csv(queries_path, encoding='utf-8-sig')
        self.rel_df = pd.read_csv(relevance_path, encoding='utf-8-sig')

        return self.docs_df, self.queries_df, self.rel_df

    def inspect_corpus(self):
        if self.docs_df is None:
            self.load_data()

        print("=" * 60)
        print("📊 АУДИТ КОРПУСА ДОКУМЕНТОВ")
        print("=" * 60)
        print(f"Всего документов в коллекции: {len(self.docs_df)}")
        print(f"Колонки датасета: {list(self.docs_df.columns)}")
        print(f"Типы данных:\n{self.docs_df.dtypes}")
        print("\nПропуски в данных (isnull):")
        print(self.docs_df.isnull().sum())

        unique_ids = self.docs_df['document_id'].nunique()
        print(f"\nУникальных document_id: {unique_ids} из {len(self.docs_df)}")
        if unique_ids == len(self.docs_df):
            print("✅ Дубликатов document_id не обнаружено.")
        else:
            print("⚠️ Обнаружены повторяющиеся document_id!")

        if 'section' in self.docs_df.columns:
            print(f"\nТематические разделы каталога:\n{self.docs_df['section'].value_counts()}")

        print(f"\nТестовых запросов: {len(self.queries_df)}")
        print(f"Открытых эталонных запросов: {len(self.rel_df)}")
        print("=" * 60 + "\n")

    def get_full_documents_texts(self):
        if self.docs_df is None:
            self.load_data()

        doc_texts = []
        for _, row in self.docs_df.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ""
            section = str(row['section']) if pd.notna(row['section']) else ""
            text = str(row['text']) if pd.notna(row['text']) else ""
            full_text = f"{title} {section} {text}".strip()
            doc_texts.append(full_text)

        doc_ids = self.docs_df['document_id'].tolist()
        return doc_texts, doc_ids


if __name__ == '__main__':
    loader = DataLoader()
    loader.inspect_corpus()
