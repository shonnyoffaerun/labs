"""
Модуль: preprocessor.py
Автор: Примакова Анна, МТ-203
Назначение: Лингвистический пайплайн обработки текста:
            очистка, токенизация, удаление стоп-слов и лемматизация.
            Обеспечивает строго одинаковую обработку документов и запросов.
"""

import os
import re
import pandas as pd
import pymorphy3

# Настройка NLTK для обхода сетевых ограничений и работы через прокси/VPN
os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'
import nltk

try:
    nltk.pathsec.ALLOW_PROXIED_FETCH = True
except AttributeError:
    pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords


class TextPreprocessor:
    """Класс предварительной обработки текста документов и поисковых запросов."""

    def __init__(self, custom_stopwords=None):
        self.morph = pymorphy3.MorphAnalyzer()
        if custom_stopwords is not None:
            self.stopwords = set(custom_stopwords)
        else:
            self.stopwords = set(stopwords.words('russian'))

    def clean_and_tokenize(self, text):
        """Приведение к нижнему регистру и извлечение буквенно-цифровых токенов."""
        if not isinstance(text, str) or pd.isna(text):
            return []
        text = text.lower()
        return re.findall(r'\b[а-яёa-z0-9]+\b', text)

    def preprocess(self, text, use_lemmatization=True, remove_stopwords=True):
        """
        Предобработка одной текстовой строки с поддержкой параметров абляции:
        - use_lemmatization: приведение токенов к нормальной форме (лемме).
        - remove_stopwords: фильтрация незначащих слов (предлоги, союзы, местоимения).
        """
        tokens = self.clean_and_tokenize(text)
        result_tokens = []

        for token in tokens:
            # 1. Первичная фильтрация стоп-слов
            if remove_stopwords and token in self.stopwords:
                continue

            # 2. Лемматизация (при включении)
            if use_lemmatization:
                parsed = self.morph.parse(token)[0]
                token = parsed.normal_form

                # Повторная проверка стоп-слов после лемматизации
                if remove_stopwords and token in self.stopwords:
                    continue

            result_tokens.append(token)

        return " ".join(result_tokens)

    def process_corpus(self, doc_texts, use_lemmatization=True, remove_stopwords=True):
        """Пакетная предобработка коллекции документов."""
        return [
            self.preprocess(text, use_lemmatization=use_lemmatization, remove_stopwords=remove_stopwords)
            for text in doc_texts
        ]

    def process_queries(self, queries, use_lemmatization=True, remove_stopwords=True):
        """Пакетная предобработка поисковых запросов по тем же правилам."""
        return [
            self.preprocess(query, use_lemmatization=use_lemmatization, remove_stopwords=remove_stopwords)
            for query in queries
        ]


if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    test_text = "Как устроены хранение и реставрация античной мраморной скульптуры?"
    print("Исходный текст:", test_text)
    print("Конфиг A (сырой):", preprocessor.preprocess(test_text, False, False))
    print("Конфиг B (+леммы):", preprocessor.preprocess(test_text, True, False))
    print("Конфиг C (+леммы, -стоп-слова):", preprocessor.preprocess(test_text, True, True))
