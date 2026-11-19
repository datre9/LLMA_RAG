from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load PDF
pdf_path = "Bc.pdf"

reader = PdfReader(pdf_path)

docs = [
    Document(
        page_content=page.extract_text() or "",
        metadata={
            "source": pdf_path,
            "page": i
        }
    )
    for i, page in enumerate(reader.pages)
]

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 3. Initialize Embeddings
embeddings = OllamaEmbeddings(model="bge-m3")

# 4. Vector Store
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_czech_db"
)

# 5. LLM
llm = ChatOllama(model="qwen2.5", temperature=0.1)

# 6. Prompt
prompt_template = """Odpovězte na otázku uživatele na základě následujícího kontextu. Pokud odpověď neznáte, řekněte, že ji v textu nemůžete najít. Odpovídejte výhradně v češtině.

Kontext:
{context}

Otázka: {question}

Odpověď:"""

prompt = PromptTemplate.from_template(prompt_template)

# Helper function to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 7. Modern RAG Chain setup
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Run query
dotaz = "Jaký byl průměrný čas testu SS pro jazyk C++?"
vysledek = rag_chain.invoke(dotaz)

print(f"\nOtázka: {dotaz}")
print(f"Odpověď: {vysledek}")