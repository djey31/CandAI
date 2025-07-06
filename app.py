import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("fr_core_news_md")

def clean_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def match_cv_offer(cv_text, offer_text):
    cv_clean = clean_text(cv_text)
    offer_clean = clean_text(offer_text)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([cv_clean, offer_clean])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    cv_tokens = set(cv_clean.split())
    offer_tokens = set(offer_clean.split())

    matched = list(cv_tokens & offer_tokens)
    missing = list(offer_tokens - cv_tokens)

    return round(score * 100, 2), matched, missing

st.title("🧠 IA Matching CV ↔ Offre d’emploi")

cv_input = st.text_area("📄 Collez ici votre CV", height=200)
offer_input = st.text_area("📢 Collez ici l'offre d’emploi", height=200)

if st.button("🔍 Analyser le matching"):
    if not cv_input.strip() or not offer_input.strip():
        st.warning("Veuillez remplir les deux champs.")
    else:
        score, matched_keywords, missing_keywords = match_cv_offer(cv_input, offer_input)

        st.subheader("✅ Résultats du matching")
        st.write(f"🔗 **Score de compatibilité : {score} %**")
        
        st.markdown("### 🟢 Mots-clés présents dans le CV")
        st.write(", ".join(matched_keywords) if matched_keywords else "Aucun mot-clé trouvé.")
        
        st.markdown("### 🔴 Mots-clés manquants dans le CV")
        st.write(", ".join(missing_keywords) if missing_keywords else "Aucun mot-clé manquant.")

        st.markdown("### 💡 Conseils personnalisés")
        if score >= 80:
            st.success("Votre profil correspond très bien à cette offre.")
        elif score >= 60:
            st.info("Profil partiellement aligné. Ajoutez les mots-clés manquants.")
        else:
            st.warning("Votre profil semble éloigné de l’offre.")
