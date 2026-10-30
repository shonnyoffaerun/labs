"""
Модуль: search_engine.py
Автор: Тюлин Константин, МТ-201
Назначение: Реализация поисковых механизмов:
    1. Базовый поиск по прямому пересечению слов
    2. Векторный поиск TF-IDF, косинусное сходство
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class WordOverlapSearcher:
    """
    Базовый поисковый движок по прямому совпадению слов (Baseline ((бизнес)) )
    Расчитывает долю одинаковых токенов и обрабатывает пустые запросы
    """

    def __init__(self, raw_documents, doc_ids):
        self.doc_ids = doc_ids
        self.doc_tokens = [
            set(re.findall(r'\b[а-яёa-z0-9]+\b', text.lower()))
            for text in raw_documents
        ]

    def search_query(self, query_text, top_k=5):
        """Поиск ТОП-*k* документов для запроса"""
        q_tokens = set(re.findall(r'\b[а-яёa-z0-9]+\b', str(query_text).lower()))
        scores = []

        for idx, d_tokens in enumerate(self.doc_tokens):
            if not q_tokens:
                score = 0.0
            else:
                score = len(q_tokens & d_tokens) / float(len(q_tokens))
            scores.append((self.doc_ids[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_search(self, queries_df, top_k=5):
        """Пакетный поиск для таблицы запросов"""
        results = []
        for _, q_row in queries_df.iterrows():
            q_id = q_row['query_id']
            top_docs = self.search_query(q_row['query'], top_k=top_k)

            for rank, (doc_id, score) in enumerate(top_docs, 1):
                results.append({
                    'query_id': q_id,
                    'rank': rank,
                    'document_id': doc_id,
                    'score': round(float(score), 4)
                })

        return pd.DataFrame(results)


class TfidfSearchEngine:
    """
    Векторная поисковая система на TF-IDF и косинусном сходстве
    Строит матрицу признаков по коллекции документов и проецирует запросы в то же векторное пространство без повторного обучения
    """

    def __init__(self, ngram_range=(1, 1)):
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
        self.doc_matrix = None
        self.doc_ids = []

    def fit(self, processed_docs, doc_ids):
        """Обучение векторизатора на документах"""
        self.doc_ids = doc_ids
        self.doc_matrix = self.vectorizer.fit_transform(processed_docs)
        return self

    def search_single(self, processed_query, top_k=5):
        """
        Поиск для одиночного запроса
        Возвращает: index, doc_id, score
        """
        if self.doc_matrix is None:
            raise ValueError("Модель TF-IDF ещё не обучена, вызовите fit()")

        q_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'index': idx,
                'document_id': self.doc_ids[idx],
                'score': round(float(similarities[idx]), 4)
            })
        return results

    def batch_search(self, processed_queries, query_ids, top_k=5):
        """
        Пакетный поиск по набору готовых запросов
        Возвращает датафрейм с колонками: query_id, rank, document_id, score
        """
        if self.doc_matrix is None:
            raise ValueError("Модель TF-IDF ещё не обучена, вызовите для начала fit()")

        query_matrix = self.vectorizer.transform(processed_queries)
        sim_matrix = cosine_similarity(query_matrix, self.doc_matrix)

        results = []
        for q_idx, q_id in enumerate(query_ids):
            scores = sim_matrix[q_idx]
            top_indices = np.argsort(scores)[::-1][:top_k]

            for rank, idx in enumerate(top_indices, 1):
                results.append({
                    'query_id': q_id,
                    'rank': rank,
                    'document_id': self.doc_ids[idx],
                    'score': round(float(scores[idx]), 4)
                })

        return pd.DataFrame(results)





"""



                    **=   :---::.
              ::--...::-::::::::::-:::
           .::==-..::.:-::-==----::::-:-.
        :--.:==+:-+++:==+==========-:::---.
      :-==.:=++-:::-*--=+++++++==+==--::-:-:
     -::-.:+++---...--==+++++++++=====--:::-::
   .-=-+=-++**=-:.:-:--==*++++*+++=====-::-:--.
  .-=:=+-.++*=:..-:*:---=*+++++++++=+====::-:-::
 .-=-.=+:-*+**...::-:-=-==++++*=*++======-:---=-
 :==:=+-:+****-:.:.::=====*++++++*++++====-:---=-
.==+++++--*****-.:-+:=--=+*++===+=+=+++++==-:=-=:.
.--++*=:.+***+:...-*:=-==++++++=+=+=*=++++=-:--=-:
:=-+**+::+***-:..:.:-=-==++*+++==*+++++++++=-:--=:
=+:-**=::****+:....:-=-==++=++===++*+*++++++---+=-
===+**-::*****+::+**-=-===*+=++++++++=+++++=---==-
=+=++*:==-*****=:.:++===+=*+=++++=++++++++===-===:           .
 =-=+*=--:=****:::**-====+*==++++=+++++++++=---=::    :::--:..
 -++-:*-:-****-:::::+--===+++=+++++*=*+++++=--==:::----=---:.:
 :=--:=*=::****::::+*--==+*++=+++++++*+**++=-::----=-=----:..
   ==-::*+=****+::+**-=+==**++++++++++-:::------====+=----..:
   ==:::=*******=:-**-+==+***+*+---------=-==---=--=+=---:..:
     ==+=+*+****+---*-+=++#**+=----==-=-------=-=-+=----:..:
       =+==***+**+*=*=++::--==-=-=-=-==---=++--+=--=---:...
         =--+=::::-----=--=-----=--+-=*=-+=++=-==-=---:..:
       ....::---=--===+=-------=--==---+=----==-----:..:
      ::....-====--=-==-==-=---=---=-==--======---:..:
       =....:----=====-=-======-====-==-======--:..:
        -:...:-==--=-=-=-=-=-======-========-::...
          ::..:--=====+==-=========-=====-::...:
            .....::---------=---===---:...::-
              -::........::::::::::...:.-
                   ::.::::::::::=


"""


"""

              -+++=                              +--
             -+***==++                        ***+*+-*
             =***#*+++++                   %#+=+*****#
              *#****+=-==-==-:.::.:..====-:-++**###*##
              +*#****+=--==-=+:=--+:-=====--+**####*##
              +###*****==--.-+-==++:-==+=:-=+**######
              -=*###*+====::+++==++.-+=-=++****+*#*+
             =-+***++==++=:::+=-*+-:.=+===++**+++**=-
            -=-=++=-====+--+=:+:=.====-=++=+++++--=-:+
           ==++*==+=+=-++::*+:=:=-+.:=+-==+==+*++*++++*
          ===+*+++++**+**-=*=----+*.+++=+=+++=**=+**+=+#
          =====::::--...::*+=--=-++#+-.:-=----=+*+++++=+
         -=-=.:+=+=*#=....-+-====**-....=*++*+:..=++=+==+
         ---:=+=====+*+-:-:#*++*##.:=::=*===++#*+-.---==-=
        :=+++*+**++-===++-:==+++=-:+*+==+=+**+++*#****++-::::
        -+-:=+++==-=++++==-==+++=:-=+=++***##**#*+=--=*+=-:...::
        :=++=**+=+***+==-==::=-:-+=-==+**##*++***++**+==--::....::
        :-=**##*+*****+==+#%*.=%##**+*#**#***+**###*+-----:::::.::
       :--=*##******+*++++*#*=+***++**#*####+++****++=-------::::-
      :--===+****++++**+***##%##**++***+++*******+====-:-=-----:-:
     ====+=-==+*####***###*####%###%#*#**#*+=+======:=+++=+-------
     -=+*=+=-++=--=***+**#*##*#*###+*#++*==---=++*++*****====---:-
    -==++===+=-=+*+--=+**######%%***++*==+==++=-====-:.:--=--:::::
    ++===-:=+*=++*++=---=*******+--::=*-++=--====:---==-:-::--::::
   ========++++=-:=+*********###****=-+*==--=-===+++:==-===-===-::
  =*==+======+*++=-:-===--=++==-=++=+**=====---+-=+++-==-=:---+--:
 +**++==--=++===--++--+++-===--::-==++===--=+++++=+++*+=*=--=-:==-
*****+=====+++=======+=++====++==-=++=++==-====+*+=:=++=
**++=+++++==--==                  +=+=++++++++++++:+
*+***+++++======                 *++==-=++==+++=+=+
 +*++++--==+=+=+                  =::==-=-==++=+==
  +++-=======--                   +:+++++:+++=-=+
   =+=+*=+++=-                     ===-=+-+++::


"""

"""

не трогайте, музейные экспонаты

"""