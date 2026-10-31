import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ------------------------------------------------------------------
# TEMA "WIKI DE MINECRAFT" (solo estilos CSS; la lógica y los inputs no cambian)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;600;700&display=swap');

:root{
  --cielo:#eef4f7;
  --pasto:#5d9b3a;
  --pasto-oscuro:#3f6b27;
  --pasto-claro:#e3efd9;
  --tierra:#866043;
  --piedra:#8b8b8b;
  --piedra-clara:#ededeb;
  --diamante:#2fb8a8;
  --tinta:#2b2b2b;
}

/* Fondo: cielo suave con el suelo de pasto y tierra abajo */
.stApp{
  background:
    linear-gradient(to bottom,
      transparent calc(100% - 64px),
      var(--pasto) calc(100% - 64px),
      var(--pasto) calc(100% - 52px),
      var(--tierra) calc(100% - 52px)),
    var(--cielo);
  background-attachment: fixed;
}
header[data-testid="stHeader"]{ background: transparent; }

/* Panel central: bloque de pasto (borde verde + franja de tierra) */
[data-testid="stMainBlockContainer"], .block-container{
  background:#fffefb;
  border-radius:6px;
  border-top:10px solid var(--pasto);
  box-shadow: inset 0 8px 0 var(--tierra), 0 18px 40px rgba(40,60,30,.14);
  margin-top:2rem;
  margin-bottom:6rem;
  padding:2.4rem 2.4rem 2.4rem !important;
  max-width:780px;
}
@media (max-width:640px){
  [data-testid="stMainBlockContainer"], .block-container{
    padding:2rem 1.1rem 1.8rem !important;
  }
}

/* Encabezado */
.mc-tag{
  display:inline-flex; align-items:center; gap:.45rem;
  font-family:'Nunito', sans-serif;
  font-weight:700; font-size:.8rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--pasto-oscuro);
  background:var(--pasto-claro);
  padding:.3rem .8rem; border-radius:4px;
}
.mc-tag svg{ width:18px; height:18px; }
.mc-title{
  font-family:'Press Start 2P', monospace;
  font-size:1.7rem; line-height:1.45;
  color:var(--tinta);
  margin:1rem 0 .7rem;
}
.mc-title span{ color:var(--pasto); }
.mc-sub{
  font-family:'Nunito', sans-serif;
  font-size:1.08rem; line-height:1.55;
  color:#5a5a55;
}
@media (max-width:640px){
  .mc-title{ font-size:1.15rem; }
}

/* Imagen: pixeles nítidos, centrada y a escala */
[data-testid="stImage"]{ margin:1.4rem auto 0; }
[data-testid="stImage"] img{
  image-rendering: pixelated;
  max-width:150px;
  display:block; margin:0 auto;
}

/* Textos del contenido */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
  font-family:'Nunito', sans-serif;
  color:var(--tinta);
  line-height:1.6;
}
h2, h3{
  font-family:'Press Start 2P', monospace !important;
  font-size:.85rem !important;
  line-height:1.7 !important;
  color:var(--pasto-oscuro) !important;
  padding-bottom:.5rem;
  border-bottom:3px solid var(--pasto-claro);
}
h1 a, h2 a, h3 a{ display:none !important; }

/* Barra lateral: piedra clara */
section[data-testid="stSidebar"]{
  background:var(--piedra-clara);
  border-right:4px solid var(--piedra);
}
section[data-testid="stSidebar"] h3{
  font-size:.65rem !important;
  border-bottom:none;
}
</style>
""", unsafe_allow_html=True)

# Título tipo wiki (con una gema de píxeles de diseño propio)
GEMA = """
<svg viewBox="0 0 16 16" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="2" width="6" height="1" fill="#9ffcf0"/>
  <rect x="3" y="3" width="10" height="1" fill="#4aedd9"/>
  <rect x="3" y="3" width="2" height="1" fill="#9ffcf0"/>
  <rect x="2" y="4" width="12" height="2" fill="#4aedd9"/>
  <rect x="2" y="4" width="3" height="1" fill="#9ffcf0"/>
  <rect x="3" y="6" width="10" height="1" fill="#2fd6c4"/>
  <rect x="4" y="7" width="8" height="1" fill="#2fd6c4"/>
  <rect x="5" y="8" width="6" height="1" fill="#1a9c90"/>
  <rect x="6" y="9" width="4" height="1" fill="#1a9c90"/>
  <rect x="7" y="10" width="2" height="1" fill="#1a9c90"/>
</svg>
"""
st.markdown(f"""
<span class="mc-tag">{GEMA} Wiki con IA</span>
<div class="mc-title">Minecraft <span>Wiki</span></div>
<div class="mc-sub">Sube una guía o manual de Minecraft en PDF y pregúntame lo que quieras: crafteos, mobs, biomas y más.</div>
""", unsafe_allow_html=True)

st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("⛏️ Este agente lee tu PDF de Minecraft y responde tus preguntas sobre él")

# Get API key from user
ke = st.text_input('🔑 Ingresa tu clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Ingresa tu clave de OpenAI para entrar al mundo.")

# PDF uploader
pdf = st.file_uploader("📜 Carga tu PDF (guía, wiki o manual de Minecraft)", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto minado: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} bloques")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("💬 ¿Qué quieres saber?")
        user_question = st.text_area(" ", placeholder="Ej: ¿Cómo encuentro diamantes?")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4o" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### 📖 Respuesta")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Algo explotó al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Ingresa tu clave de OpenAI para entrar al mundo.")
else:
    st.info("Carga un PDF para comenzar la aventura.")
