CO POTREBUJETE:
nainstalovat ollama

pouzivaji se tyto model (pokud vam to neutahne, tak si muzete stahnout menis)
    - ollama pull qwen2.5
    - ollama pull bge-m3

stahnout python knihovny (snad jsem na zadne nezapomel)
    pip install langchain langchain-community langchain-ollama chromadb pypdf fontTools pymupdf sentence_transformers


JAK TO ROZJET
    - nejdrive spustit indexer.py -> vytvori embedingy vsech souboru ve slozce pdfs
        - staci spustit, jen kdyz se obsah slozky zmeni
    - spustit rag.py (dole si muzete zmenit otazku)