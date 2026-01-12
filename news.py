from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Page configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .category-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .world-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sports-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .business-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .tech-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .info-box {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    

    </style>
""", unsafe_allow_html=True)


def text_preprocessing(text):
    """Preprocess text for classification"""
    v1 = "".join([i for i in text.lower() if i not in punctuation])
    v2 = [lemmatizer.lemmatize(i, 'v') for i in v1.split() if i not in stop]
    return " ".join([lemmatizer.lemmatize(i, 'r') for i in v2 if i not in stop])

# Load model


@st.cache_resource
def load_model():
    try:
        with open('news.pkl', 'rb') as ob:
            data = pickle.load(ob)
        return data
    except FileNotFoundError:
        st.error(
            "⚠️ Model file 'news.pkl' not found. Please ensure the file is in the same directory.")
        return None


# Header
st.markdown("""
    <div class="header-container">
        <h1 style="margin: 0; font-size: 2.5rem;">📰 News Article Classifier</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Classify news into categories instantly
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 About This App")
    st.markdown("""
    This intelligent classifier analyzes news articles and categorizes them into:
    
    - 🌍 **World News**
    - 🏏 **Sports**
    - 📈 **Business**
    - 🤖 **Science & Technology**
    
    ---
    
    ### 🔍 How to Use
    1. Enter the article title 
    2. Paste or type the article description
    3. Click "Classify Article" button
    4. Get instant results!
    
    ---
    
    ### 💡 Tips
    - Longer descriptions give better results
    - Include key details from the article
    - Try different news sources
    """)


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter Article Details")

    # Input fields
    title = st.text_input(
        "Article Title",
        placeholder="Enter the headline or title of the news article...",
        # help="The title helps provide context but is not required for classification"
    )

    des = st.text_area(
        "Article Description *",
        placeholder="Paste or type the main content of the news article here...",
        height=200,
        # help="Enter the article text you want to classify. More text provides better accuracy."
    )

    # Character counter
    if des:
        char_count = len(des)
        word_count = len(des.split())
        st.caption(f"📝 {char_count} characters | {word_count} words")

with col2:
    st.markdown("### 📌 Quick Guide")
    st.markdown("""
    <div class="info-box">
        <h4 style="margin-top: 0;">Category Examples:</h4>
        <p><strong>🌍 World:</strong> International events, politics, diplomacy</p>
        <p><strong>🏏 Sports:</strong> Games, tournaments, athletes</p>
        <p><strong>📈 Business:</strong> Markets, companies, economy</p>
        <p><strong>🤖 Sci/Tech:</strong> Innovation, research, gadgets</p>
    </div>
    """, unsafe_allow_html=True)

# Classify button
st.markdown("<br>", unsafe_allow_html=True)
check = st.button('🔍 Classify Article', use_container_width=True)

# Classification logic
if check:
    if not des:
        st.warning("⚠️ Please enter the article description to classify.")
    else:
        data = load_model()
        if data is not None:
            with st.spinner('🔄 Analyzing article...'):
                try:
                    # Simulate processing time
                    import time
                    time.sleep(0.5)

                    res = data['model'].predict([des])[0]

                    st.markdown("---")
                    st.markdown("### 🎯 Classification Result")

                    # Display results with styled cards
                    if res == 1:
                        st.markdown("""
                            <div class="category-card world-card">
                                <h2 style="margin: 0;">🌍 World News</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    This article is related to international events, global politics, or world affairs.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    elif res == 2:
                        st.markdown("""
                            <div class="category-card sports-card">
                                <h2 style="margin: 0;">🏏 Sports</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    This article is related to sports, athletics, games, or competitions.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    elif res == 3:
                        st.markdown("""
                            <div class="category-card business-card">
                                <h2 style="margin: 0;">📈 Business</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    This article is related to business, finance, markets, or economy.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    elif res == 4:
                        st.markdown("""
                            <div class="category-card tech-card">
                                <h2 style="margin: 0;">🤖 Science & Technology</h2>
                                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                                    This article is related to science, technology, innovation, or research.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.warning(
                            "⚠️ This article doesn't clearly fit into any of the predefined categories.")

                    # Display article preview
                    if title or des:
                        with st.expander("📄 View Article Preview"):
                            if title:
                                st.markdown(f"**Title:** {title}")
                            st.markdown(
                                f"**Description:** {des[:300]}{'...' if len(des) > 300 else ''}")

                except Exception as e:
                    st.error(
                        f"❌ An error occurred during classification: {str(e)}")
# Footer
st.markdown("---")
# st.markdown("""
#     <div style="text-align: center; color: #666; padding: 1rem;">
#         <p>Made by Ebin Raj</p>
#     </div>
# """, unsafe_allow_html=True)
