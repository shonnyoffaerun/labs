"""
Модуль: app.py
Интерактивный веб-интерфейс «Виртуальная арт-галерея» на базе Streamlit.
Использует модули команды:
- data_loader (Кравченко Дмитрий)
- preprocessor (Примакова Анна)
- search_engine (Тюлин Константин)
- evaluator (Шагиев Даниил)
- main (Лонщаков Александр)
"""

import os
import sys
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_loader import DataLoader
from preprocessor import TextPreprocessor
from search_engine import TfidfSearchEngine

st.set_page_config(
    page_title="Академия Искусств | Скульптура и Живопись",
    page_icon="🏛️",
    layout="wide"
)

# Стилизация под виртуальный музей
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(30, 25, 20, 0.45), rgba(20, 17, 14, 0.55)),
                          url('https://img.magnific.com/premium-photo/grand-art-museum-gallery-with-sculptures-paintings-displayed-pedestals_1261973-3278.jpg?semt=ais_hybrid&w=740&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f7f3ec;
        font-family: 'Georgia', serif;
    }
    h1, h2, h3 {
        color: #fffaf0 !important;
        font-family: 'Georgia', serif;
        letter-spacing: 0.8px;
        text-shadow: 0 2px 6px rgba(0,0,0,0.8);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(25, 21, 17, 0.92);
        border-right: 1px solid #5a4735;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #f0e6d8 !important;
        font-family: sans-serif;
    }
    .stTextInput input {
        background-color: rgba(35, 29, 24, 0.95);
        color: #fffaf0;
        border: 1px solid #8c6d4a;
        border-radius: 6px;
        padding: 12px;
        font-family: 'Georgia', serif;
    }
    .stTextInput input:focus {
        border-color: #d4af37;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.6);
    }
    .stTextInput label {
        background-color: rgba(247, 243, 236, 0.88);
        padding: 6px 12px;
        border-radius: 6px;
    }
    .stTextInput label p, [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-weight: 600;
        font-family: 'Georgia', serif;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label p {
        color: #f0e6d8 !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(247, 243, 236, 0.88);
    }
    [data-testid="stHeader"] *, [data-testid="stToolbar"] *,
    [data-testid="stAppDeployButton"] *, [data-testid="stMainMenu"] * {
        color: #000000 !important;
    }
    .art-banner {
        padding: 28px;
        background: linear-gradient(135deg, rgba(70, 56, 43, 0.88) 0%, rgba(25, 21, 17, 0.94) 100%);
        border-left: 6px solid #d4af37;
        border: 1px solid #6b533f;
        border-radius: 8px;
        margin-bottom: 30px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.7);
    }
    .art-banner h1 {
        margin: 0;
        font-size: 30px;
        color: #fffaf0;
    }
    .art-banner p {
        margin: 10px 0 0 0;
        color: #e2d2bc;
        font-size: 15px;
        font-family: sans-serif;
    }
    .exhibit-card {
        background: linear-gradient(145deg, rgba(32, 26, 21, 0.92) 0%, rgba(22, 18, 14, 0.96) 100%);
        border: 1px solid #6b533f;
        border-left: 4px solid #d4af37;
        padding: 20px;
        border-radius: 6px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
        transition: all 0.3s ease;
    }
    .exhibit-card:hover {
        border-color: #d4af37;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.2);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_and_preprocessor():
    loader = DataLoader()
    docs_df, _, _ = loader.load_data()
    doc_texts, doc_ids = loader.get_full_documents_texts()
    preprocessor = TextPreprocessor()
    return docs_df, doc_texts, doc_ids, preprocessor


try:
    docs_df, doc_texts, doc_ids, preprocessor = get_data_and_preprocessor()
except Exception as e:
    st.error(f"Не удалось инициализировать данные: {e}")
    st.stop()

st.markdown("""
    <div class="art-banner">
        <h1>🏛️ Галерея Скульптуры и Живописи | Поисковый Каталог</h1>
        <p>Интеллектуальная система поиска по классическим произведениям искусства и академическим изваяниям</p>
    </div>
""", unsafe_allow_html=True)

# Боковая панель
st.sidebar.header("🎨 Настройки экспозиции")
st.sidebar.markdown("Выберите конфигурацию лингвистического анализа:")

config_option = st.sidebar.selectbox(
    "Конфигурация модели:",
    (
        "Конфигурация A (Сырой текст, униграммы)",
        "Конфигурация B (+ Лемматизация)",
        "Конфигурация C (+ Лемматизация, - Стоп-слова) [Рекомендуется]",
        "Конфигурация D (+ Лемматизация, - Стоп-слова, Биграммы)"
    ),
    index=2
)

use_lemmas = False
remove_stops = False
ngram_range = (1, 1)

if "B" in config_option:
    use_lemmas = True
elif "C" in config_option:
    use_lemmas = True
    remove_stops = True
elif "D" in config_option:
    use_lemmas = True
    remove_stops = True
    ngram_range = (1, 2)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Параметры текущей конфигурации:**
- Лемматизация: **{'Вкл' if use_lemmas else 'Выкл'}**
- Стоп-слова: **{'Удаляются' if remove_stops else 'Остаются'}**
- N-граммы: **{ngram_range}**
""")

st.sidebar.markdown("""
---
**Авторы проекта:**
- Кравченко Д. (Загрузка данных)
- Примакова А. (Препроцессинг)
- Тюлин К. (Поисковый движок)
- Шагиев Д. (Метрики и оценка)
- Лонщаков А. (Точка входа и веб-приложение)
""")


@st.cache_resource
def build_search_engine(use_lemmas, remove_stops, ngram_range):
    prepared_docs = preprocessor.process_corpus(
        doc_texts,
        use_lemmatization=use_lemmas,
        remove_stopwords=remove_stops
    )
    engine = TfidfSearchEngine(ngram_range=ngram_range)
    engine.fit(prepared_docs, doc_ids)
    return engine


with st.spinner("✨ Инициализация векторного пространства экспозиции..."):
    engine = build_search_engine(use_lemmas, remove_stops, ngram_range)

st.subheader("🔍 Поиск по коллекции статуй, барельефов и картин")
user_query = st.text_input(
    "Введите поисковый запрос (например: «мраморная скульптура античного героя» или «картина маслом»):",
    value="мраморная статуя"
)

if user_query:
    prep_query = preprocessor.preprocess(user_query, use_lemmatization=use_lemmas, remove_stopwords=remove_stops)
    top_results = engine.search_single(prep_query, top_k=5)

    st.markdown("### 🏆 ТОП-5 релевантных экспонатов")

    for res in top_results:
        rank = res['rank']
        score = res['score']
        row = docs_df[docs_df['document_id'] == res['document_id']].iloc[0]

        st.markdown(f"""
            <div class="exhibit-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background-color: #5a4735; color: #fffaf0; padding: 3px 12px; border-radius: 4px; font-weight: bold; font-size: 13px; font-family: sans-serif;">Ранг #{rank}</span>
                    <span style="color: #d4af37; font-weight: bold; font-size: 14px; font-family: sans-serif;">Коэффициент сходства: {score:.4f}</span>
                </div>
                <h3 style="margin: 0 0 8px 0; color: #fffaf0; font-family: 'Georgia', serif;">{row['title']}</h3>
                <div style="font-size: 13px; color: #d4c5b0; margin-bottom: 10px; font-family: sans-serif;"><b>Раздел:</b> {row['section']} &nbsp;|&nbsp; <b>ID экспоната:</b> {row['document_id']}</div>
                <p style="margin: 0; color: #f0e6d8; font-size: 14px; line-height: 1.6; font-family: sans-serif;">{row['text']}</p>
            </div>
        """, unsafe_allow_html=True)
