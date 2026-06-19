# 🤖 Assistente Inteligente para PDFs (RAG)

Sistema de **Retrieval-Augmented Generation (RAG)** que combina busca semântica com geração de respostas via IA local. Permite fazer perguntas sobre seus documentos PDF e obter respostas contextualizadas usando Ollama (llama3), FAISS e embeddings locais.

## 👨‍💻 Stacks:

- pypdf: Extrai texto dos documentos PDF
- langchain_ollama: Integra com o modelo de IA Ollama (rodando localmente)
- faiss: Busca rápida usando embeddings (vetores matemáticos)
- pickle: Armazena metadados dos documentos

## 🔄 Como funciona

```
 PDFs (pypdf)
    ↓
 Chunks (2000 caracteres + 400 overlap)
    ↓
 Embeddings (nomic-embed-text via Ollama)
    ↓
 FAISS Index (vector_store/)
    ↓
 Pergunta do usuário
    ↓
 Busca de 15 chunks mais relevantes (com filtro >= 0.3 score)
    ↓
 Contexto + Histórico → LLM (llama3 via Ollama)
    ↓
 Resposta personalizada em português
```

## 🏗️ Arquitetura

| Componente | Tecnologia | Descrição |
|---|---|---|
| **Loader** | `pypdf` | Extrai texto de PDFs |
| **Splitter** | Custom | Divide em chunks com sobreposição inteligente |
| **Embeddings** | `nomic-embed-text` | Converte texto em vetores (384-dim) |
| **Vector Store** | FAISS | Busca rápida (O(1)) de chunks similares |
| **LLM** | `llama3` (7B) | Gera respostas inteligentes |
| **Persistência** | FAISS + pickle | Reutiliza índice entre execuções |

## 📋 Estrutura do Projeto

```
.
├── RAG.py                    #  Código principal (420+ linhas bem comentadas)
├── requirements.txt          #  Dependências Python
├── README.md                 #  Este arquivo
├── data/                     #  Pasta para seus PDFs (git-ignored)
└── vector_store/             #  Índice FAISS (git-ignored, criado automaticamente)
    ├── index.faiss           # Índice de busca
    └── metadata.pkl          # Metadados dos chunks
```

## 🚀 Setup

### 📋 Pré-requisitos

- **Python 3.11+** (testado com 3.11 e 3.12)
- **Ollama** rodando localmente: [ollama.ai](https://ollama.ai)
  - Baixe e instale Ollama
  - Pull dos modelos:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```
  - Verifique se está rodando: `ollama serve` (porta padrão: 11434)

### 1. Criar e ativar ambiente virtual

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

Se houver conflitos, use:
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Adicionar PDFs

Crie uma pasta `data/` e coloque seus PDFs lá:

```
data/
├── documento1.pdf
├── documento2.pdf
└── curriculum.pdf
```

**Nota:** A pasta `data/` é ignorada pelo git (dados locais).

### 4. Executar

```powershell
python RAG.py
```

**Primeira execução:**
- Lê todos os PDFs de `data/`
- Divide em ~2000 caracteres por chunk (com 400 de sobreposição)
- Gera embeddings (pode levar 5-10 minutos)
- Cria índice FAISS em `vector_store/`
- Inicia o chat

**Execuções seguintes:**
- Carrega índice existente (~1 segundo)
- Inicia o chat imediatamente 

## 💬 Uso

```
========================================================
🤖 RAG Assistant — digite 'sair' para encerrar
========================================================

🔍 Buscando informações...
   📊 Relevância média: 0.523 | Chunks encontrados: 12

💬 Você: Quais são as principais habilidades?

🤖 Assistente: Com base nos documentos, as principais habilidades 
incluem: Python, machine learning, análise de dados...

Fontes utilizadas:
   → curriculum.pdf | página 1
   → curriculum.pdf | página 2

💬 Você: sair
👋 Encerrando. Até mais!
```

### Comandos

- **Digite qualquer pergunta** em português natural
- **'sair'/'exit'/'quit'** para encerrar
- **Histórico** das últimas 10 respostas é mantido no contexto

## ⚙️ Parâmetros Ajustáveis

Edite diretamente em `RAG.py`:

### Topo do arquivo (configuração global)

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama |
| `LLM_MODEL` | `llama3` | Modelo Ollama para gerar respostas |
| `EMBED_MODEL` | `nomic-embed-text` | Modelo para embeddings (vetores) |
| `PDF_DIR` | `Path("data")` | Pasta com PDFs |
| `VECTOR_STORE_PATH` | `Path("vector_store")` | Pasta do índice FAISS |

### Na função `split_documents()`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `chunk_size` | `2000` | Caracteres por chunk (~500 palavras) |
| `overlap` | `400` | Sobreposição entre chunks (20%) |

### Na função `retrieve()`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `k` | `15` | Chunks a recuperar por busca |
| `score >= 0.3` | `0.3` | Filtro mínimo de relevância (cosine similarity) |

### Na função `main()`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `temperature` | `0` | Criatividade do LLM (0=determinístico, 1=criativo) |

### No template `PROMPT_TEMPLATE`

Controla como o modelo IA responde. Atual: flexível, honesto sobre limitações.

## 🔧 Troubleshooting

### ❌ "Ollama não está respondendo"
```powershell
# Verifique se Ollama está rodando
ollama serve

# Teste a conexão
curl http://localhost:11434/api/tags
```

### ❌ "Modelo não encontrado"
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### ❌ "Vector store não encontrado"
Delete a pasta `vector_store/` e rode novamente:
```powershell
rmdir vector_store -r
python RAG.py
```

### ❌ "Nenhum chunk com relevância suficiente encontrado"
- **Causa:** Score de similaridade < 0.3 para todos os chunks
- **Solução:** Reduza o threshold em `retrieve()` (ex: `if score >= 0.2`)
- **Ou:** Aumente `k` de 15 para 20+ para trazer mais candidatos

### ❌ "Conflito de dependências"
```powershell
pip install --upgrade pip setuptools wheel
pip install --force-reinstall -r requirements.txt
```

## ⚡ Performance

| Métrica | Valor | Descrição |
|---|---|---|
| **Embedding** | <100ms | `nomic-embed-text` é leve (~220MB) |
| **Busca FAISS** | O(1) | Busca rápida após indexação |
| **Resposta LLM** | 5-30s | Depende da complexidade |
| **1ª execução** | 5-15 min | Gera embeddings de todos os chunks |
| **Execuções seguintes** | <1s | Carrega índice do disco |

## 🎯 Mudanças Recentes (v2)

✅ **Busca Melhorada**
- Aumentado de 8 para **15 chunks** por busca
- Filtro de relevância: apenas chunks com **score >= 0.3**
- Debug visual: mostra relevância média e quantidade encontrada

✅ **Prompt Inteligente**
- Menos restritivo (flexível ao invés de "APENAS")
- Reconhece quando informação não está disponível
- Melhor direcionamento para respostas concisas

✅ **Chamada LLM Simplificada**
- Uma única tentativa de invocação (versão estável)
- Suporta múltiplos formatos de resposta

✅ **Debug Melhorado**
- Feedback visual (emojis) durante busca
- Exibição de relevância média dos chunks
- Avisos claros quando nenhum chunk é relevante

## 📌 Limitações Atuais

- Recupera máximo de 15 chunks por pergunta (ajustável)
- Histórico de conversa resetado a cada execução
- Sem suporte a imagens em PDFs
- Índice não atualiza automaticamente (deletar `vector_store/` para reindexar)
- Sem cache de embeddings individuais

## 🚀 Melhorias Futuras

- [ ] Persistência de histórico entre sessões
- [ ] Multiple retrieval strategies (BM25 + semantic)
- [ ] Filtros por metadata (data, página, arquivo)
- [ ] Suporte a PDFs com imagens (OCR)
- [ ] API REST (FastAPI)
- [ ] Interface web (Streamlit)
- [ ] Sincronização incremental de índice
- [ ] Cache inteligente de embeddings

## 📖 Licença

Projeto desenvolvido para fins educacionais.

---

## 🎬 Quick Start (TL;DR)

```bash
# 1. Instalar Ollama e puxar modelos
ollama pull llama3 && ollama pull nomic-embed-text

# 2. Setup Python
python -m venv venv && .\venv\Scripts\Activate
pip install -r requirements.txt

# 3. Adicionar PDFs
mkdir data
# copiar arquivos .pdf para a pasta data/

# 4. Rodar
python RAG.py

# 5. Fazer perguntas!
# Digite suas perguntas em português natural
```