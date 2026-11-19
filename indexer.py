from pathlib import Path
import shutil
import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Config
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
DB_DIR = BASE_DIR / "chroma_czech_db"


# 0. Delete existing database
if DB_DIR.exists():
    shutil.rmtree(DB_DIR)
    print(f"Deleted old vector store: {DB_DIR}")


# 1. Load PDFs
pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

docs = []

for pdf_path in pdf_paths:
    with pymupdf.open(pdf_path) as pdf:
        title = pdf.metadata.get("title") or pdf_path.stem
        total_pages = len(pdf)

        for page in pdf:
            text = page.get_text("text", sort=True)

            if not text.strip():
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page.number + 1,
                        "total_pages": total_pages,
                        "title": title,
                    },
                )
            )

print(f"Loaded {len(docs)} pages from {len(pdf_paths)} PDF(s)")


# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

chunks = text_splitter.split_documents(docs)

counters = {}

for c in chunks:
    key = (c.metadata["source"], c.metadata["page"])
    counters[key] = counters.get(key, 0) + 1
    c.metadata["chunk"] = counters[key]

print(f"Created {len(chunks)} chunks")


# 3. Embeddings
embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://localhost:11434")


# 4. Create vector store
vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)

# 5. Add documents to vector store in batches
batch_size = 128

for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    vector_db.add_documents(batch)

    print(f"Added batch {i // batch_size + 1} " f"({len(batch)} chunks)")


print("Indexing completed!")