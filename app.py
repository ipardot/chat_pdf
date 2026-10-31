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
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

:root{
  --grass:#5d9b3a;
  --grass-dark:#3f6b27;
  --dirt:#7a4e24;
  --stone:#7f7f7f;
  --stone-dark:#4a4a4a;
  --stone-light:#c6c6c6;
  --parchment:#f3e9d2;
  --parchment-edge:#d8c9a3;
  --ink:#3b2f1e;
  --gold:#fcdb05;
  --diamond:#4aedd9;
}

/* Fondo: cielo + franja de pasto y tierra abajo */
.stApp{
  background:
    linear-gradient(to bottom,
      transparent calc(100% - 90px),
      var(--grass) calc(100% - 90px),
      var(--grass) calc(100% - 74px),
      var(--grass-dark) calc(100% - 74px),
      var(--grass-dark) calc(100% - 66px),
      var(--dirt) calc(100% - 66px)),
    linear-gradient(#6ec0ff 0%, #bfe6ff 100%);
  background-attachment: fixed;
  font-family: 'VT323', monospace;
}
header[data-testid="stHeader"]{ background: transparent; }

/* Panel central: página de wiki (pergamino) */
[data-testid="stMainBlockContainer"], .block-container{
  background: var(--parchment);
  border: 4px solid #2b2b2b;
  box-shadow: inset 0 0 0 4px var(--parchment-edge), 8px 8px 0 rgba(0,0,0,.35);
  margin-top: 2rem;
  margin-bottom: 7rem;
  padding: 1.5rem 2.2rem 2.2rem !important;
  max-width: 900px;
}

/* Banner del título */
.mc-hero{
  display:flex; align-items:center; justify-content:center; gap:1.2rem;
  background: var(--grass);
  border: 4px solid #1f1f1f;
  box-shadow: inset 4px 4px 0 #7fc257, inset -4px -4px 0 var(--grass-dark), 5px 5px 0 rgba(0,0,0,.3);
  padding: 1.1rem 1rem;
  margin: .5rem 0 1.4rem;
  text-align:center;
}
.mc-hero svg{ width:56px; height:56px; image-rendering:pixelated; flex:none; }
.mc-title{
  font-family:'Press Start 2P', monospace;
  font-size:1.5rem; color:#fff;
  text-shadow:3px 3px 0 #2b4a1a;
  line-height:1.4;
}
.mc-sub{
  font-family:'VT323', monospace;
  font-size:1.6rem; color:#fffbe0;
  text-shadow:2px 2px 0 #2b4a1a;
  margin-top:.35rem;
}
@media (max-width:640px){
  .mc-title{ font-size:1rem; }
  .mc-sub{ font-size:1.25rem; }
  .mc-hero svg{ width:36px; height:36px; }
}

/* Tipografía del contenido */
h2, h3{
  font-family:'Press Start 2P', monospace !important;
  color: var(--ink) !important;
  font-size: .95rem !important;
  line-height:1.6 !important;
  border-bottom: 4px solid var(--parchment-edge);
  padding-bottom:.4rem;
}
h1 a, h2 a, h3 a{ display:none !important; }
.stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"]{
  font-family:'VT323', monospace !important;
  font-size:1.3rem;
  color: var(--ink);
}

/* Barra lateral: piedra */
section[data-testid="stSidebar"]{
  background: repeating-conic-gradient(var(--stone) 0% 25%, #6e6e6e 0% 50%) 0 0 / 32px 32px;
  border-right: 4px solid #1f1f1f;
}
section[data-testid="stSidebar"] *{
  color:#fff !important;
  text-shadow:2px 2px 0 #2b2b2b;
  font-family:'VT323', monospace !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  font-family:'Press Start 2P', monospace !important;
  font-size:.7rem !important;
  line-height:1.7 !important;
  border-bottom:none;
}

/* Imagen con marco */
[data-testid="stImage"] img{
  image-rendering: pixelated;
  border: 4px solid #2b2b2b;
  box-shadow: 0 0 0 4px var(--parchment-edge), 6px 6px 0 4px rgba(0,0,0,.3);
}

/* Campos de entrada: solo apariencia */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea{
  background:#262626 !important;
  color:#fff !important;
  font-family:'VT323', monospace !important;
  font-size:1.3rem !important;
  border:4px solid #000 !important;
  border-radius:0 !important;
  box-shadow: inset 3px 3px 0 #0a0a0a, inset -3px -3px 0 #5a5a5a !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
  outline:none !important;
  border-color: var(--diamond) !important;
}
[data-testid="stTextInput"] > div > div,
[data-testid="stTextArea"] > div > div{
  border-radius:0 !important; border:none !important; background:transparent !important;
}

/* Cargador de archivos */
[data-testid="stFileUploader"] section{
  background:#3a3a3a !important;
  border:4px dashed var(--stone-light) !important;
  border-radius:0 !important;
}
[data-testid="stFileUploader"] section *{ color: var(--stone-light) !important; }

/* Botones: bloque de piedra */
.stButton > button,
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"]{
  font-family:'Press Start 2P', monospace !important;
  font-size:.6rem !important;
  color:#fff !important;
  background: var(--stone) !important;
  border:4px solid #000 !important;
  border-radius:0 !important;
  box-shadow: inset 3px 3px 0 var(--stone-light), inset -3px -3px 0 var(--stone-dark) !important;
  text-shadow:2px 2px 0 #3f3f3f;
  transition:none !important;
}
.stButton > button:hover,
[data-testid="stFileUploader"] button:hover{
  background:#8f9fff !important;
  box-shadow: inset 3px 3px 0 #c8d0ff, inset -3px -3px 0 #4a56a8 !important;
}

/* Alertas: bloques de mineral */
[data-testid="stAlert"]{
  border-radius:0 !important;
  border:4px solid #000 !important;
  box-shadow:4px 4px 0 rgba(0,0,0,.3);
}
[data-testid="stAlert"] *{
  color:#fff !important;
  font-family:'VT323', monospace !important;
  font-size:1.25rem !important;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){ background:#2f4f6f !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){ background:#1f6b35 !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){ background:#8a6d00 !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){ background:#8a1111 !important; }

/* Código / traceback */
pre, code{
  font-family:'VT323', monospace !important;
  background:#111 !important;
  color:#17dd62 !important;
  border-radius:0 !important;
}

/* Scrollbar cuadrada */
::-webkit-scrollbar{ width:14px; height:14px; }
::-webkit-scrollbar-track{ background:#1a1a1a; }
::-webkit-scrollbar-thumb{ background:var(--stone); border:3px solid #000; }
</style>
""", unsafe_allow_html=True)

# Título tipo wiki (con gemas de píxeles de diseño propio)
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
<div class="mc-hero">
  {GEMA}
  <div>
    <div class="mc-title">MINECRAFT WIKI</div>
    <div class="mc-sub">Pregúntame lo que quieras saber de Minecraft</div>
  </div>
  {GEMA}
</div>
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
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
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
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
