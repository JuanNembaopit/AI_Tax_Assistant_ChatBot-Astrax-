import streamlit as st
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
import os
import re

# --- Set Page Config First ---
st.set_page_config(
    page_title="Astrax - Asisten Pajak DJP",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Memilih Tema
theme_choice = st.sidebar.selectbox("Pilih Tema:", ["Light ☀️", "Dark 🌙"])

# Menggunakan tema yang diinginkan
if theme_choice == "Light ☀️":
    primary_color = "#2A5C82"
    secondary_color = "#F0F4F8"
    accent_color = "#FF6B35"
    sidebar_bg = "#ffffff"
    sidebar_text = "#000000"
    selected_bg = "#2A5C82"
    selected_text = "#ffffff"
    text_color = "#ffffff"
    bg_color = "#f8f9fa"
    header = "#F0F4F8"
elif theme_choice == "Dark 🌙":
    primary_color = "#ffffff"
    secondary_color = "#1E1E1E"
    accent_color = "#FF6B35"
    sidebar_bg = "#0e1117"
    sidebar_text = "#ffffff"
    selected_bg = "#f39c12"
    selected_text = "#000000"
    text_color = "#000000"
    bg_color = "#252422"
    header = "#1E1E1E"


# --- Custom CSS ---
st.markdown(f"""
<style>
    :root {{
        --primary: {primary_color};
        --secondary: {secondary_color};
        --accent: {accent_color};
        --text: {text_color};
    }}

    .stApp {{
        background-color: {bg_color};
    }}

    .header {{
        padding: 1rem 0;
        border-bottom: 2px solid var(--primary);
        margin-bottom: 2rem;
    }}

    .chat-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }}

    .user-message {{
        background: var(--secondary);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #DEE2E6;
    }}

    .assistant-message {{
        background: var(--primary);
        color: var(--text);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #DEE2E6;
    }}

    .pdf-ref {{
        color: var(--primary);
        border-left: 3px solid var(--accent);
        padding-left: 1rem;
        margin-top: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:

    selected = option_menu(
        "Menu",
        ["Chatbot", "Tentang", "Kontak"],
        icons=["chat-dots", "info-circle", "telephone"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": sidebar_bg},
            "icon": {"color": accent_color, "font-size": "25px"},
            "nav-link": {"color": sidebar_text, "font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": selected_bg, "color": selected_text},
        }
    )

# Load Environment Variables
load_dotenv()
MONGODB_URI = os.environ.get("MONGO_URI")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

# MongoDB Connection with Caching
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        client.server_info()
        return client['Astrax_db']['Astrax']
    except Exception as e:
        st.error(f"⚠️ Gagal terhubung ke database: {str(e)}")
        st.stop()

collection = init_mongodb()

# Initialize Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Vector Store Configuration
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name='vector_index'
)

# Model Definition
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_KEY, temperature=0)

# Create Prompt
template = """
Anda adalah asisten digital resmi bernama M-Tax yang bekerja untuk Direktorat Jenderal Pajak Indonesia. 
Tugas Anda adalah membantu wajib pajak memahami informasi perpajakan, khususnya yang berkaitan dengan layanan digital seperti M-Tax, e-filing, e-billing, dan sistem pajak online lainnya.

Jawaban Anda harus:
- Disampaikan dalam Bahasa Indonesia,
- Ramah, sopan, dan profesional,
- Berdasarkan informasi dari konteks yang tersedia,
- Fokus pada topik perpajakan dan layanan digital DJP.

Jika pertanyaan tidak relevan dengan perpajakan atau layanan resmi DJP, sampaikan dengan sopan bahwa Anda hanya dapat menjawab pertanyaan seputar pajak dan layanan digital pemerintah Indonesia.

---

{context}

**Pertanyaan dari pengguna:**
{question}

**Jawaban M-Tax:**
"""

prompt = PromptTemplate(
    template=template, 
    input_variables=[
        'context', 
        'question'])

# Retrieval Chain with Error Handling
@st.cache_resource
def create_qa_chain():
    try:
        return RetrievalQA.from_chain_type(
            llm=llm,chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
                ),
            chain_type_kwargs={"prompt": prompt}
        )
    except Exception as e:
        st.error(f"⚠️ Gagal inisialisasi sistem: {str(e)}")
        st.stop()

qa = create_qa_chain()

# --- Streamlit UI ---
if selected == "Chatbot":
    with st.container():
        col1, col2 = st.columns([4, 4])
        with col1:
            st.image("djp.png", width=1000)
        with col2:
            st.markdown("""
            <div class="header">
                <h1 style="color: var(--primary); margin: 0;">Astrax</h1>
                <p style="color: #666; margin: 0;">Asisten Pajak Digital Direktorat Jenderal Pajak</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-container">
        <div style="background: var(--secondary); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            🤖 Tanyakan apa saja tentang:
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 0.5rem;">
                <span style="padding: 0.3rem; border-radius: 5px;">✔️ Pelaporan Individu</span>
                <span style="padding: 0.3rem; border-radius: 5px;">✔️ e-Faktur</span>
                <span style="padding: 0.3rem; border-radius: 5px;">✔️ SPT Tahunan</span>
                <span style="padding: 0.3rem; border-radius: 5px;">✔️ Pelaporan Perusahaan</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Selamat datang! Saya Astrax , asisten virtual Direktorat Jenderal Pajak. Bagaimana saya bisa membantu Anda hari ini?"
        }]

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-container">
                <div class="user-message">
                    😀 <strong>Anda</strong><br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-container">
                <div class="assistant-message">
                    🤖 <strong>Astrax</strong><br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Input Chat
    question = st.chat_input("Tulis pertanyaan pajak Anda di sini... (Contoh: Bagaimana cara reset password DJP Online?)")

    if question:
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Prepare Answer
        with st.spinner("Mencari informasi..."):
            try:
                result = qa.invoke({"query": question})
                answer = result['result'].strip()

                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"Terjadi kesalahan sistem: {str(e)}\nSilakan coba lagi atau hubungi 1500200"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
        st.rerun()

elif selected == "Tentang":
    st.markdown("### 📟 Tentang Astrax")
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
        Astrax (Asisten Pajak) adalah chatbot AI yang membantu pengguna memahami dan menjawab pertanyaan umum terkait sistem pelaporan pajak Indonesia. 
        Dibuat menggunakan LangChain dan LLM OpenAI, bot ini memberikan jawaban untuk pertanyaan yang sering diajukan seperti cara mengajukan pajak (e-Filing), 
        pertanyaan terkait pelaporan (SPT), terkait akun (kehilangan kata sandi), dan banyak lagi. 
        Chatbot ini dirancang untuk mengurangi kebingungan, meningkatkan aksesibilitas, dan mendukung transformasi pajak digital Indonesia yang sedang berlangsung.
    </div>        
    """, unsafe_allow_html=True)

elif selected == "Kontak":
    st.markdown("### 📩 Kontak")
    st.markdown("""
    **Jika Anda memiliki pertanyaan atau ingin berkolaborasi, hubungi kami melalui email berikut:**

    - 📧 [galuh.adika@gmail.com](mailto:galuh.adika@gmail.com) — *Data Analyst + Project Management*
    - 📧 [adeindrar@gmail.com](mailto:adeindrar@gmail.com) — *Data Engineer*
    - 📧 [eldimuhamads@gmail.com](mailto:eldimuhamads@gmail.com) — *Data Science*
    - 📧 [juannembaopit13@gmail.com](mailto:juannembaopit13@gmail.com) — *Data Science*
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #666; font-size: 0.9rem;">
    <hr style="margin: 2rem 0;">
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
        Percakapan Anda aman dan terenkripsi
    </div>
    <p style="margin: 0.5rem 0;">Direktorat Jenderal Pajak Republik Indonesia</p>
    <p style="margin: 0;">© 2025 Sistem Layanan Pajak Digital</p>
</div>
""", unsafe_allow_html=True)