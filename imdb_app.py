import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="IMDB Shows Recommendation App", layout="centered")

st.title("IMDB / Netflix Shows Content-Based Recommender")
st.write(
    "Bu uygulama, dizi/film veri setindeki tür, yönetmen, oyuncu ve başlık özelliklerini birleştirerek benzer içerikler önerir."
)

@st.cache_data
def load_data():
    df = pd.read_csv("imdb_processed.csv", low_memory=False)
    df["genre"] = df["genre"].fillna("")
    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")
    return df

try:
    df = load_data()
    
    def create_soup(x):
        return (
            str(x["genre"])
            + " "
            + str(x["director"])
            + " "
            + str(x["cast"])
            + " "
            + str(x["title"])
        )

    df["soup"] = df.apply(create_soup, axis=1)

    @st.cache_data
    def compute_similarity(data):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(data["soup"])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    cosine_sim = compute_similarity(df)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

    def get_recommendations(title):
        if title not in indices:
            return None
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return df["title"].iloc[movie_indices]

    st.subheader("İçerik Öneri Paneli")
    unique_titles = df["title"].dropna().unique()
    selected_show = st.selectbox("Bir Dizi veya Film Seçin", unique_titles)

    if st.button("Öneri Al"):
        recommendations = get_recommendations(selected_show)
        if recommendations is not None:
            st.write(f"'{selected_show}' için önerilen benzer yapımlar:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.warning("Seçilen yapım için öneri bulunamadı.")

except Exception as e:
    st.error(f"Veri yüklenirken veya işlenirken bir hata oluştu: {e}")