"""
SISTEMA RAG (Retrieval-Augmented Generation)

SOBRE: Sistema inteligente que combina busca em documentos com geração de respostas.
- O usuário faz uma pergunta
- O sistema busca trechos relevantes nos PDFs
- O modelo de IA gera uma resposta personalizada baseada nesses trechos.

STACK:
- pypdf: Extrai texto dos documentos PDF
- langchain_ollama: Integra com o modelo de IA Ollama (rodando localmente)
- faiss: Busca rápida usando embeddings (vetores matemáticos)
- pickle: Armazena metadados dos documentos

FLUXO:
1. Carrega PDFs da pasta 'data/'
2. Divide o texto em chunks pequenos com sobreposição
3. Converte cada chunk em um vetor numérico (embedding)
4. Armazena os vetores em um índice FAISS para busca rápida
5. Quando o usuário faz uma pergunta, busca os chunks mais relevantes
6. Envia os chunks + pergunta para o Ollama gerar uma resposta
"""

from pathlib import Path
import os
import pickle
from typing import List, Dict
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ===== CONFIGURAÇÕES DO SISTEMA =====
# Local onde o Ollama está rodando
OLLAMA_BASE_URL = "http://localhost:11434"

# Modelo de IA para gerar respostas
LLM_MODEL = "llama3"

# Modelo para converter texto em vetores/embeddings 
EMBED_MODEL = "nomic-embed-text"

# Pasta onde os arquivos PDF estão armazenados
PDF_DIR = Path("data")

# Pasta onde o índice de busca FAISS será armazenado 
VECTOR_STORE_PATH = Path("vector_store")


def load_pdfs(pdf_dir: Path) -> List[Dict]:
    """
    FUNÇÃO: Carrega todos os PDFs de uma pasta
    
    ENTRADA: Caminho da pasta com PDFs
    SAÍDA: Lista de dicionários com estrutura: {'source': nome_do_arquivo, 'page': número_página, 'text': texto_extraído}
    
    IMPORTANTE:
    - Procura por TODOS os arquivos .pdf na pasta
    - Extrai texto de TODAS as páginas
    - Se um PDF não tiver texto, a página pode vir vazia
    """
    # Procura por todos os PDFs na pasta (glob = busca por padrão)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    # Se não encontrou nenhum PDF, lança um erro
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em '{pdf_dir}/'")

    pages = []
    # Itera sobre cada arquivo PDF encontrado
    for pdf_path in pdf_files:
        # Cria um leitor de PDF usando pypdf
        reader = PdfReader(str(pdf_path))
        # Percorre cada página do PDF
        for i, page in enumerate(reader.pages):
            # Extrai o texto da página (se não conseguir, usa string vazia)
            text = page.extract_text() or ""
            # Armazena a página com suas metadatas (origem, número, texto)
            pages.append({"source": str(pdf_path), "page": i, "text": text})

    # Mostra quantas páginas foram carregadas no total
    print(f"Carregados {len(pages)} páginas de {len(pdf_files)} PDF(s)")
    return pages


def split_documents(pages: List[Dict], chunk_size: int = 2000, overlap: int = 400) -> List[Dict]:
    """
    FUNÇÃO: Divide os textos das páginas em pedaços menores (chunks)
    
    POR QUE? Os modelos de IA têm limite de tokens. Melhor buscar pequenos pedaços relevantes
    do que tentar processar o documento inteiro de uma vez.
    
    PARÂMETROS:
    - chunk_size: tamanho de cada pedaço em caracteres (2000 = ~500 palavras)
    - overlap: quanto os pedaços se sobrepõem (400 = 20% de sobreposição)
    
    SOBREPOSIÇÃO:
    Se um conceito importante cai na borda de um chunk, a sobreposição garante
    que ele apareça nos dois chunks, melhorando a busca semantica.
    """
    chunks = []
    # Para cada página do documento
    for p in pages:
        # Remove caracteres especiais de quebra de linha
        text = p["text"].replace("\r", "")
        
        # Se a página está vazia, pula para a próxima
        if not text.strip():
            continue
        
        # Inicializa as variáveis para dividir o texto
        start = 0
        L = len(text)
        
        # Enquanto não chegou ao final do texto
        while start < L:
            # Define onde o chunk vai terminar (ou no chunk_size, ou no final)
            end = min(start + chunk_size, L)
            # Extrai o pedaço de texto
            chunk_text = text[start:end].strip()
            
            # Se o chunk tem conteúdo, armazena ele com suas metadatas
            if chunk_text:
                chunks.append({"source": p["source"], "page": p["page"], "text": chunk_text})
            
            # Move para o próximo chunk (chunk_size - overlap garante a sobreposição)
            start += chunk_size - overlap

    # Mostra quantos chunks foram criados
    print(f"Dividido em {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks: List[Dict], embeddings: OllamaEmbeddings, index_path: Path = VECTOR_STORE_PATH) -> None:
    """
    FUNÇÃO: Converte os chunks em vetores (embeddings) e cria um índice de busca FAISS
    
    COMO FUNCIONA:
    1. Pega cada chunk de texto
    2. Envia para o modelo OllamaEmbeddings converter em um vetor de números (~384 números)
    3. Normaliza os vetores (deixa todos com tamanho 1 - cosine similarity)
    4. Armazena em um índice FAISS para busca rápida usando dot product
    5. Guarda os metadados em um arquivo pickle
    
    TEMPO: Pode demorar bastante! (minutos) porque gera embeddings para cada chunk
    
    SAÍDA: Cria 2 arquivos em vector_store/
    - index.faiss: o índice de busca (vetores)
    - metadata.pkl: informações sobre cada chunk (source, page, text)
    """
    # Cria a pasta vector_store se não existir
    os.makedirs(index_path, exist_ok=True)
    
    # Extrai apenas os textos dos chunks (descarta metadatas temporariamente)
    texts = [c["text"] for c in chunks]
    
    # Informa ao usuário que vai demorar
    print("Gerando embeddings (pode levar um tempo)...")
    
    # Envia os textos para o modelo gerar embeddings
    # Cada texto se torna um vetor de números
    vectors = embeddings.embed_documents(texts)
    # Converte a lista de vetores em um array numpy (formato que FAISS entende)
    arr = np.array(vectors).astype('float32')
    
    # NORMALIZA os vetores para usar COSINE SIMILARITY
    # Isso significa que a busca vai medir o ÂNGULO entre vetores (0° = identico, 90° = diferente)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)  # Calcula o tamanho de cada vetor
    norms[norms == 0] = 1.0  # Evita divisão por zero
    arr = arr / norms  # Divide cada vetor pelo seu tamanho (normaliza)

    # Obtém o número de dimensões dos vetores (ex: 384)
    d = arr.shape[1]
    
    # Cria um índice FAISS que usa INNER PRODUCT (funciona com vetores normalizados)
    # IndexFlatIP = procura exaustiva, mas muito rápida para documentos menores
    index = faiss.IndexFlatIP(d)
    
    # Adiciona todos os vetores ao índice
    index.add(arr)

    # Salva o índice FAISS em disco para reutilizar depois (não precisa recalcular)
    faiss.write_index(index, str(index_path / 'index.faiss'))
    
    # Salva os metadados (source, page, text) em um arquivo pickle
    # Quando buscar, retorna os índices do FAISS e recupera o texto aqui
    with open(index_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(chunks, f)

    # Mostra sucesso e quantidade de vetores armazenados
    print(f"Index salvo em '{index_path}/' com {index.ntotal} vetores")


def load_vector_store(index_path: Path = VECTOR_STORE_PATH):
    """
    FUNÇÃO: Carrega o índice FAISS e os metadados do disco
    
    SAÍDA: Tupla com (index, metadata)
    - index: objeto FAISS pronto para fazer buscas
    - metadata: lista com os chunks originais (source, page, text)
    """
    # Define os caminhos dos arquivos que vão ser carregados
    idx_file = index_path / 'index.faiss'
    meta_file = index_path / 'metadata.pkl'
    
    # Verifica se os arquivos existem (se não, significa que não construiu o índice ainda)
    if not idx_file.exists() or not meta_file.exists():
        raise FileNotFoundError("Vector store não encontrado. Execute a construção primeiro.")
    
    # Carrega o índice FAISS do arquivo
    index = faiss.read_index(str(idx_file))
    
    # Carrega os metadados (chunks originais) do arquivo pickle
    with open(meta_file, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


def retrieve(index, metadata: List[Dict], embeddings: OllamaEmbeddings, query: str, k: int = 15) -> List[Dict]:
    """
    FUNÇÃO: Busca os chunks mais relevantes para uma pergunta (query)
    
    PARÂMETRO k: Quantos chunks retornar (padrão 15 para melhor cobertura)
    
    SAÍDA: Lista dos k chunks mais similares à pergunta
    
    FUNCIONAMENTO:
    1. Converte a pergunta em um vetor (embedding) usando o mesmo modelo
    2. Normaliza o vetor
    3. Busca no índice FAISS os vetores mais próximos (usando cosine similarity)
    4. Retorna os chunks correspondentes do metadata com filtro de relevância
    """
    # Converte a pergunta (string) em um vetor numérico
    qv = np.array(embeddings.embed_query(query)).astype('float32')
    
    # Normaliza o vetor da pergunta (mesmo que fez com os chunks)
    qv = qv / (np.linalg.norm(qv) + 1e-10)
    
    # BUSCA NO FAISS: retorna os k vetores mais similares
    # D = distâncias (similarity scores)
    # I = índices dos chunks no metadata
    D, I = index.search(np.expand_dims(qv, axis=0), k)
    
    # Recupera os chunks do metadata usando os índices retornados
    results = []
    scores = []
    for idx, score in zip(I[0], D[0]):
        # Verifica se o índice é válido (segurança)
        if idx < 0 or idx >= len(metadata):
            continue
        # Filtra por score mínimo de relevância (cosine similarity >= 0.3)
        if score >= 0.3:
            results.append(metadata[idx])
            scores.append(score)
    
    # DEBUG: Mostra os scores de relevância encontrados
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"   Relevância média: {avg_score:.3f} | Chunks encontrados: {len(results)}")
    else:
        print(f"   Nenhum chunk com relevância suficiente encontrado!")
    
    return results


# 📋 TEMPLATE DO PROMPT
# Este é o "prompt system" que controla como o modelo de IA vai agir
# As chaves {context}, {chat_history}, {question} serão preenchidas dinamicamente
PROMPT_TEMPLATE = (
    "Você é um assistente especializado nos documentos fornecidos.\n"
    "Use o contexto abaixo para responder à pergunta da forma mais completa possível.\n"
    "Se a resposta não estiver claramente no contexto, indique isso de forma honesta.\n"
    "Responda sempre em português.\n"
    "Seja conciso mas informativo.\n\n"
    "Contexto dos documentos:\n{context}\n\n"  # Será substituído pelos chunks relevantes
    "Histórico da conversa:\n{chat_history}\n\n"  # Histórico das perguntas/respostas anteriores
    "Pergunta do usuário: {question}\n\nResposta:"  # A pergunta atual do usuário
)


def llm_generate(llm: ChatOllama, prompt: str) -> str:
    """
    FUNÇÃO: Envia um prompt para o modelo Ollama e obtém a resposta
    
    ENTRADA: prompt (string com a pergunta e contexto)
    SAÍDA: resposta do modelo (string)
    
    ERRO: Se falhar, lança uma exceção informando para verificar o Ollama
    """
    # Chamar o modelo com um string direto (método padrão do langchain_ollama)
    out = llm.invoke(prompt)
    
    # Extrai o texto da resposta
    if hasattr(out, 'content'):
        return out.content
    if isinstance(out, str):
        return out
    if hasattr(out, 'text'):
        return out.text
    
    # Se o formato for diferente, converte para string
    return str(out)


def print_sources(docs: List[Dict]):
    """
    FUNÇÃO: Mostra ao usuário quais documentos/páginas foram usados para gerar a resposta
    """
    print("\nFontes utilizadas:")
    
    # Usa um set para evitar mostrar a mesma fonte duas vezes
    seen = set()
    for d in docs:
        # Cria uma chave única (source + page)
        key = (d.get('source'), d.get('page'))
        
        # Se já mostrou essa fonte, pula
        if key in seen:
            continue
        
        # Marca essa fonte como já mostrada
        seen.add(key)
        
        # Mostra o nome do arquivo e o número da página (em formato legível para o usuário)
        print(f"   → {Path(d.get('source')).name} | página {d.get('page') + 1}")


def chat_loop(index, metadata, embeddings: OllamaEmbeddings, llm: ChatOllama):
    """
    FUNÇÃO: Loop principal de conversa com o usuário
    
    FLUXO EM CADA MENSAGEM:
    1. Usuário digita uma pergunta
    2. Sistema busca 4 chunks relevantes
    3. Cria um prompt com contexto + histórico + pergunta
    4. Envia para o Ollama gerar uma resposta
    5. Mostra a resposta e as fontes usadas
    6. Guarda a conversa no histórico
    """
    # Armazena o histórico de conversas (perguntas + respostas)
    history = []
    
    # Mostra a tela de boas-vindas
    print("=" * 55)
    print("🤖 RAG Assistant — digite 'sair' para encerrar")
    print("=" * 55)
    
    # Loop infinito até o usuário digitar 'sair'
    while True:
        # Lê a pergunta do usuário
        question = input("\n💬 Você: ").strip()
        
        # Se a pergunta está vazia, pede de novo
        if not question:
            continue
        
        # Se o usuário quer sair
        if question.lower() in ('sair', 'exit', 'quit'):
            print("👋 Encerrando. Até mais!")
            break

        # ===== ETAPA 1: BUSCA =====
        # Busca chunks mais relevantes para a pergunta
        print(f"🔍 Buscando informações...")
        sources = retrieve(index, metadata, embeddings, question)
        
        # ===== ETAPA 2: MONTAGEM DO CONTEXTO =====
        # Verifica se encontrou chunks relevantes
        if not sources:
            print("❌ Não encontrei informações relevantes nos documentos para essa pergunta.")
            continue
        
        # Junta os textos dos chunks com separador
        # Isso dá ao LLM mais informação para gerar respostas melhores
        context = "\n---\n".join([s['text'] for s in sources])
        
        # Pega o histórico das últimas 10 conversas para dar contexto ao modelo
        # Isso permite que o modelo entenda referências de perguntas anteriores
        chat_history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-10:]])

        # ===== ETAPA 3: PREENCHIMENTO DO PROMPT =====
        # Preenche o template com contexto real, histórico e pergunta
        prompt = PROMPT_TEMPLATE.format(context=context, chat_history=chat_history_text, question=question)
        
        # ===== ETAPA 4: GERAÇÃO DA RESPOSTA =====
        # Envia o prompt ao Ollama e obtém a resposta
        answer = llm_generate(llm, prompt)

        # ===== ETAPA 5: APRESENTAÇÃO =====
        # Mostra a resposta ao usuário
        print(f"\n🤖 Assistente: {answer}")
        
        # Mostra quais documentos/páginas foram usados
        print_sources(sources)
        
        # ===== ETAPA 6: ARMAZENAMENTO =====
        # Guarda essa conversa no histórico para referências futuras
        history.append((question, answer))


def main():
    """
    FUNÇÃO: Ponto de entrada do programa
    
    FLUXO GERAL:
    1. Inicializa o modelo de embeddings e o LLM
    2. Verifica se o índice FAISS já existe
       - SE EXISTE: carrega do disco (rápido)
       - SE NÃO EXISTE: carrega PDFs, cria chunks, gera embeddings, cria índice
    3. Inicia o loop de conversa com o usuário
    """
    # Inicializa o modelo para gerar embeddings (convertendo texto em vetores)
    # OllamaEmbeddings usa o modelo local rodando no localhost:11434
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Inicializa o modelo de linguagem para gerar respostas
    # temperature=0 significa respostas determinísticas (sem aleatoriedade)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    # ===== VERIFICA SE O ÍNDICE JÁ EXISTE =====
    if VECTOR_STORE_PATH.exists() and (VECTOR_STORE_PATH / 'index.faiss').exists():
        # ATALHO: Índice já existe, apenas carrega do disco
        print("Carregando índice existente...")
        index, metadata = load_vector_store(VECTOR_STORE_PATH)
    else:
        # PRIMEIRA VEZ: Precisa construir o índice
        print("Construindo índice pela primeira vez...")
        
        # 1. Carrega os PDFs
        pages = load_pdfs(PDF_DIR)
        
        # 2. Divide em chunks (com novo tamanho: 2000 caracteres + 400 overlap)
        chunks = split_documents(pages)
        
        # 3. Gera embeddings e cria índice FAISS
        build_vector_store(chunks, embeddings, VECTOR_STORE_PATH)
        
        # 4. Carrega o índice que acabou de criar
        index, metadata = load_vector_store(VECTOR_STORE_PATH)
        
        print("\nÍndice criado e pronto para usar!")

    # ===== INICIA O CHAT =====
    # Passa o índice, metadados, embeddings e LLM para o loop de conversa
    print(f"\n{len(metadata)} chunks carregados. Pronto para responder perguntas!\n")
    chat_loop(index, metadata, embeddings, llm)


# ===== EXECUÇÃO =====
# Este trecho garante que o main() só é executado quando o arquivo é rodado diretamente
# (não quando é importado como módulo em outro arquivo)
if __name__ == '__main__':
    main()