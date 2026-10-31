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
# TEMA MINECRAFT (solo estilos CSS; la lógica y los inputs no cambian)
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

:root{
  --grass:#5d9b3a;
  --grass-dark:#3f6b27;
  --dirt:#8b5a2b;
  --dirt-dark:#6b4423;
  --stone:#7f7f7f;
  --stone-dark:#4a4a4a;
  --stone-light:#c6c6c6;
  --obsidian:#1a1025;
  --gold:#fcdb05;
  --emerald:#17dd62;
  --redstone:#c81414;
  --diamond:#4aedd9;
  --text:#ffffff;
}

/* Fondo: bloques de tierra pixelados + franja de césped arriba */
.stApp{
  background:
    linear-gradient(var(--grass) 0, var(--grass) 14px, var(--grass-dark) 14px, var(--grass-dark) 22px, transparent 22px),
    repeating-conic-gradient(var(--dirt) 0% 25%, var(--dirt-dark) 0% 50%) 0 0 / 48px 48px;
  image-rendering: pixelated;
  color: var(--text);
  font-family: 'VT323', monospace;
  font-size: 1.25rem;
}
header[data-testid="stHeader"]{ background: transparent; }

/* Panel central tipo "inventario" */
[data-testid="stMainBlockContainer"], .block-container{
  background: rgba(20, 12, 8, .82);
  border: 4px solid #000;
  box-shadow: inset 4px 4px 0 var(--stone-dark), inset -4px -4px 0 #2b2b2b, 8px 8px 0 rgba(0,0,0,.45);
  margin-top: 2.5rem;
  margin-bottom: 2rem;
  padding: 2rem 2.5rem !important;
}

/* Tipografía */
h1, h2, h3{
  font-family: 'Press Start 2P', monospace !important;
  color: #fff !important;
  text-shadow: 3px 3px 0 #3f3f3f;
  line-height: 1.5 !important;
  letter-spacing: .5px;
}
h1{ font-size: 1.5rem !important; color: var(--gold) !important; text-shadow: 3px 3px 0 #5a4a00; }
h2, h3{ font-size: 1rem !important; }
h1 a, h2 a, h3 a{ display:none !important; }

p, li, label, span, div[data-testid="stMarkdownContainer"]{
  font-family: 'VT323', monospace !important;
  font-size: 1.3rem;
  color: var(--text);
}
[data-testid="stMarkdownContainer"] p{ text-shadow: 2px 2px 0 rgba(0,0,0,.6); }

/* Barra lateral: piedra */
section[data-testid="stSidebar"]{
  background:
    repeating-conic-gradient(var(--stone) 0% 25%, #6e6e6e 0% 50%) 0 0 / 32px 32px;
  border-right: 4px solid #000;
  box-shadow: inset -4px 0 0 var(--stone-dark);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  color: #fff !important;
  font-size: .8rem !important;
  text-shadow: 2px 2px 0 #2b2b2b;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{ text-shadow: 2px 2px 0 #2b2b2b; }

/* Imagen con marco de ítem */
[data-testid="stImage"] img{
  image-rendering: pixelated;
  border: 4px solid #000;
  box-shadow: 0 0 0 4px var(--stone-light), 6px 6px 0 4px rgba(0,0,0,.5);
}

/* Campos de entrada: solo apariencia (inventario oscuro con bisel) */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea{
  background: #1e1e1e !important;
  color: var(--gold) !important;
  font-family: 'VT323', monospace !important;
  font-size: 1.3rem !important;
  border: 4px solid #000 !important;
  border-radius: 0 !important;
  box-shadow: inset 3px 3px 0 #0a0a0a, inset -3px -3px 0 #5a5a5a !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
  outline: none !important;
  border-color: var(--diamond) !important;
}
[data-testid="stTextInput"] > div > div,
[data-testid="stTextArea"] > div > div{ border-radius: 0 !important; border: none !important; background: transparent !important; }

/* Cargador de archivos: mesa de crafteo */
[data-testid="stFileUploader"] section{
  background: #3a3a3a !important;
  border: 4px dashed var(--stone-light) !important;
  border-radius: 0 !important;
  box-shadow: inset 3px 3px 0 #1a1a1a;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span{ color: var(--stone-light) !important; }

/* Botones: bloque de piedra con bisel */
.stButton > button,
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"]{
  font-family: 'Press Start 2P', monospace !important;
  font-size: .6rem !important;
  color: #fff !important;
  background: var(--stone) !important;
  border: 4px solid #000 !important;
  border-radius: 0 !important;
  box-shadow: inset 3px 3px 0 var(--stone-light), inset -3px -3px 0 var(--stone-dark) !important;
  text-shadow: 2px 2px 0 #3f3f3f;
  transition: none !important;
}
.stButton > button:hover,
[data-testid="stFileUploader"] button:hover{
  background: #8f9fff !important;
  box-shadow: inset 3px 3px 0 #c8d0ff, inset -3px -3px 0 #4a56a8 !important;
}

/* Alertas: bloques de mineral */
[data-testid="stAlert"]{
  border-radius: 0 !important;
  border: 4px solid #000 !important;
  box-shadow: 4px 4px 0 rgba(0,0,0,.5);
  font-family: 'VT323', monospace !important;
}
[data-testid="stAlert"] *{ color:#fff !important; font-family:'VT323', monospace !important; font-size:1.25rem !important; }
[data-testid="stAlertContentInfo"]{ background: var(--stone-dark); }
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){ background:#2f4f6f !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){ background:#1f6b35 !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){ background:#8a6d00 !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){ background:#8a1111 !important; }

/* Bloque de traceback / código */
pre, code{
  font-family: 'VT323', monospace !important;
  background:#111 !important;
  color: var(--emerald) !important;
  border-radius: 0 !important;
}

/* Scrollbar cuadrada */
::-webkit-scrollbar{ width: 14px; height: 14px; }
::-webkit-scrollbar-track{ background:#1a1a1a; }
::-webkit-scrollbar-thumb{ background: var(--stone); border: 3px solid #000; }
</style>
""", unsafe_allow_html=True)

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
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
