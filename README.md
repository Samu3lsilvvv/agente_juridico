# ⚖️ Assistente Jurídico

Aplicação web de inteligência artificial para análise de documentos jurídicos, combinando recuperação semântica (RAG) com um modelo de linguagem de grande escala (LLM) para responder perguntas com base em um contrato pré-carregado.

---

## 🧠 Como funciona

O sistema implementa o padrão **RAG (Retrieval-Augmented Generation)**:

1. Na inicialização, o `Contrato.pdf` é carregado automaticamente e indexado no ChromaDB
2. O documento é segmentado em chunks e transformado em vetores de embeddings semânticos
3. Ao receber uma pergunta, o sistema recupera os trechos mais relevantes por similaridade de cosseno
4. Os trechos recuperados são enviados como contexto ao LLM, que gera uma resposta fundamentada no documento

Isso garante que as respostas estejam **ancoradas no conteúdo real do contrato**, reduzindo alucinações e aumentando a rastreabilidade das informações.

---

## 🛠️ Tecnologias utilizadas

| Camada | Tecnologia |
|---|---|
| Interface | [Streamlit](https://streamlit.io/) |
| LLM | `openai/gpt-oss-20b` via [Groq API](https://groq.com/) |
| Orquestração | [LangChain](https://www.langchain.com/) |
| Embeddings | `sentence-transformers/msmarco-bert-base-dot-v5` (HuggingFace) |
| Banco vetorial | [ChromaDB](https://www.trychroma.com/) |
| Leitura de PDF | PyPDF |

---

## 📁 Estrutura do projeto

```
.
├── agente_juridico.py   # App principal com pipeline RAG e PDF pré-carregado
├── Contrato.pdf         # Base de conhecimento do assistente
├── requirements.txt     # Dependências do projeto
└── README.md
```

---

## ⚙️ Instalação e execução

### Pré-requisitos

- Python 3.13+
- Conda (recomendado) ou virtualenv
- Chave de API da [Groq](https://console.groq.com/)

### 1. Criar e ativar o ambiente virtual

```bash
conda create --name assistente-juridico python=3.13
conda activate assistente-juridico
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Executar a aplicação

```bash
streamlit run agente_juridico.py --server.port 8502
```

> ⚠️ Caso encontre problemas de conexão, verifique se antivírus ou firewall estão bloqueando as portas.

---

## 🔑 Configuração da API Key

Ao abrir o aplicativo, insira sua **GROQ API Key** na barra lateral. Você pode obter uma gratuitamente em [console.groq.com/keys](https://console.groq.com/keys).

---

## 💬 Exemplos de perguntas

O contrato já está carregado como base de conhecimento. Exemplos de perguntas:

- *Qual é o objeto do contrato?*
- *Quem é a CONTRATANTE?*
- *Qual o valor total do contrato?*
- *Quantos profissionais a CONTRATADA deve alocar?*
- *Qual a multa em caso de rescisão imotivada pela CONTRATANTE?*
- *Qual o prazo da obrigação de confidencialidade após o término do contrato?*

---

## 📄 Licença

Este projeto está disponível sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
