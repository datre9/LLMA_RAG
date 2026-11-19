from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# Config
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_czech_db"


# 1. Embeddings
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")


# 2. Load existing vector store
if not DB_DIR.exists():
    raise FileNotFoundError("Vector database not found. Run indexer.py first.")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


# 3. LLM
llm = ChatOllama(model="qwen2.5", temperature=0.1, base_url="http://localhost:11434")


# 4. Prompt & Formatting
prompt_template = """Odpovězte na otázku uživatele na základě následujícího kontextu.
Pokud odpověď neznáte, řekněte, že ji v textu nemůžete najít.
Odpovídejte výhradně v češtině.

Kontext:
{context}

Otázka: {question}

Odpověď:"""

prompt = PromptTemplate.from_template(prompt_template)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['source']}, s. {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


# 5. Retriever
base_retriever = vector_db.as_retriever(
    search_type="mmr", search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.7}
)


# 6. Reranker
reranker_model = CrossEncoder("BAAI/bge-reranker-base")


def rerank_documents(inputs):
    question = inputs["question"]
    retrieved_docs = inputs["docs"]

    if not retrieved_docs:
        return []

    pairs = [[question, doc.page_content] for doc in retrieved_docs]

    scores = reranker_model.predict(pairs).tolist()

    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    TOP_K_FINAL = 5
    reranked_docs = [doc for doc, score in scored_docs[:TOP_K_FINAL]]

    for doc, score in scored_docs[:TOP_K_FINAL]:
        print(
            f"Reranked doc: {doc.metadata['source']}, "
            f"page {doc.metadata['page']}, score {score:.4f}"
        )

    return reranked_docs


# 7. RAG chain
retrieval_chain = RunnableParallel(
    docs=base_retriever, question=RunnablePassthrough()
) | RunnablePassthrough().assign(docs=lambda x: rerank_documents(x))

answer_chain = (
    {
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain = retrieval_chain.assign(answer=answer_chain)

# Usage
question = "Jaky je proces prevodu NKA na DKA?"
result = rag_chain.invoke(question)

print(f"\nOtázka: {question}")
print(f"Odpověď:\n{result['answer']}")

print("\nZdroje (po re-rankingu):")

for d in result["docs"]:
    print(
        f"  - {d.metadata['source']}, "
        f"strana {d.metadata['page']}, "
        f"chunk {d.metadata['chunk']}"
    )