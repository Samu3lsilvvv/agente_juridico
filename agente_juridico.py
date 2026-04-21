import os
import pathlib
import streamlit.components.v1 as components

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Evita conflitos de paralelismo com o tokenizador HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Caminho do PDF fixo, relativo ao diretório do script
PDF_PATH = pathlib.Path(__file__).parent / "Contrato.pdf"

# --- Configuração da página e sidebar ---
st.set_page_config(page_title = "Assistente Jurídico", page_icon = "⚖️", layout = "wide")

# Remove o padding excessivo do topo para alinhar o título com o header da sidebar
st.markdown("""
<style>
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
    }
</style>
""", unsafe_allow_html = True)

# --- Modo noturno ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

DARK_CSS = """
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #1c2128 !important;
        color: #fafafa !important;
        border-color: #30363d !important;
    }
    .stButton > button {
        background-color: #21262d !important;
        color: #fafafa !important;
        border-color: #30363d !important;
    }
    .stButton > button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }
    .stMarkdown, .stCaption, label, p, h1, h2, h3 {
        color: #fafafa !important;
    }
    [data-testid="stInfo"] {
        background-color: #1c2128 !important;
        color: #fafafa !important;
    }
    [data-testid="stWarning"] {
        background-color: #2d2007 !important;
        color: #e3b341 !important;
    }
    [data-testid="stLinkButton"] a {
        background-color: #21262d !important;
        color: #fafafa !important;
        border-color: #30363d !important;
    }
    [data-testid="stLinkButton"] a:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }
</style>
"""

if st.session_state.dark_mode:
    st.markdown(DARK_CSS, unsafe_allow_html = True)

# JS move o container do botão para o header nativo (mesma altura de Deploy e ⋮)
# Usa window.parent para acessar o DOM da página a partir do iframe do components.html
toggle_color = "#fafafa" if st.session_state.dark_mode else "rgb(49, 51, 63)"
components.html(f"""
<script>
  (function() {{
    function moveToggle() {{
      const doc = window.parent.document;
      const btn = Array.from(doc.querySelectorAll('button')).find(b =>
        b.textContent.includes('Modo claro') || b.textContent.includes('Modo escuro')
      );
      if (btn) {{
        const container = btn.closest('[data-testid="stElementContainer"]');
        if (container) {{
          container.style.position = 'fixed';
          container.style.top = '0.4rem';
          container.style.right = '6.5rem';
          container.style.zIndex = '999999';
          btn.style.background = 'none';
          btn.style.border = 'none';
          btn.style.boxShadow = 'none';
          btn.style.color = '{toggle_color}';
          btn.style.fontSize = '0.875rem';
          btn.style.padding = '0.25rem 0.75rem';
          btn.style.cursor = 'pointer';
          btn.style.borderRadius = '0.375rem';
          btn.style.fontFamily = 'inherit';
          btn.style.fontWeight = '400';
          return true;
        }}
      }}
      return false;
    }}
    if (!moveToggle()) {{
      const obs = new MutationObserver(function(_, o) {{
        if (moveToggle()) o.disconnect();
      }});
      obs.observe(window.parent.document.body, {{ childList: true, subtree: true }});
    }}
  }})();
</script>
""", height = 0)

toggle_label = "\u2600\ufe0f Modo claro" if st.session_state.dark_mode else "\U0001f319 Modo escuro"
if st.button(toggle_label, key = "dark_toggle"):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Coloque aqui sua GROQ API Key e pressione Enter", type = "password")
    st.link_button("🔑 Obter API Key no Groq Console", "https://console.groq.com/keys")
    st.divider()
    st.subheader("Instruções")
    st.write("1) Informe sua chave no campo acima.\n2) Digite sua pergunta ou dúvida.\n3) Clique em Enviar.")
    st.info("Aviso: a IA pode cometer erros. Verifique fatos críticos.")

st.title("⚖️ Assistente Jurídico")
st.caption("Modelo: openai/gpt-oss-20b via Groq + LangChain")

if not api_key:
    st.warning("Informe a GROQ API Key na barra lateral para continuar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(model = "openai/gpt-oss-20b", temperature = 0.2, max_tokens = 1024)


@st.cache_resource(show_spinner = "Carregando base de conhecimento…")
def cria_banco_vetorial(pdf_path: str) -> Chroma:
    """Carrega o PDF fixo, divide em chunks e indexa os embeddings no ChromaDB.
    Executado uma única vez graças ao cache do Streamlit."""
    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/msmarco-bert-base-dot-v5")
    vectordb = Chroma.from_documents(documents = chunks, embedding = embeddings)
    return vectordb


if not PDF_PATH.exists():
    st.error(f"Arquivo não encontrado: {PDF_PATH}")
    st.stop()

vectordb = cria_banco_vetorial(str(PDF_PATH))

# --- Retriever: busca os 3 chunks mais relevantes por similaridade semântica ---
retriever = vectordb.as_retriever(search_kwargs = {"k": 3})

# --- Prompt do assistente jurídico ---
system_block = """Você é um assistente jurídico que responde usando estritamente o conteúdo do PDF fornecido quando possível.
Se a resposta não estiver no PDF, diga que não encontrou no documento e ofereça passos de verificação.
Formate a resposta com: Resumo, Fundamentação (com citações de trechos entre aspas) e Próximos passos.
Se houver conflito entre o PDF e conhecimento externo, priorize o PDF e sinalize a divergência."""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content = system_block),
        ("human", "Pergunta: {question}\n\nContexto do PDF:\n{context}\n\nResponda de forma sucinta, técnica e didática.")
    ]
)


def formata_docs(docs):
    """Formata os documentos recuperados, incluindo referência de página."""
    out = []
    for d in docs:
        meta = d.metadata or {}
        where = f"p.{meta.get('page', '?')}"
        out.append(f'[{where}] "{d.page_content.strip()[:800]}{"…" if len(d.page_content) > 800 else ""}"')
    return "\n\n".join(out)


# --- Interface de pergunta e resposta ---
pergunta = st.text_area("Escreva sua pergunta jurídica", height = 120, placeholder = "Ex.: Quais cláusulas tratam de rescisão e multas?")

col1, col2 = st.columns([1, 1])
with col1:
    btn = st.button("Perguntar")

if btn:
    if not pergunta.strip():
        st.warning("Digite uma pergunta antes de enviar.")
        st.stop()

    # Pipeline RAG: recupera contexto do PDF → monta prompt → invoca o LLM
    rag_pipeline = RunnableParallel(
        context = retriever | formata_docs,
        question = RunnablePassthrough()
    ) | qa_prompt | llm | StrOutputParser()

    with st.spinner("Gerando resposta…"):
        answer = rag_pipeline.invoke(pergunta)

    st.markdown("### Resposta")
    st.write(answer)
