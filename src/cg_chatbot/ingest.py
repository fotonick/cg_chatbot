# Ingest docs into persistant ChromaDB

import os
import shutil
import sys


def main():
    args = sys.argv[1:]

    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings

    md_text = open("docs/cully-grove-declaration-and-bylaws-recorded-ocr.md").read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    # First split by Markdown section headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    md_header_splits = markdown_splitter.split_text(md_text)

    # Some sections are longer than the LLM's max context length. Furthermore, we combine up to four
    # documents at a time as part of the final generation task. So split sections further.
    chunk_size = 1536
    chunk_overlap = 64
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(md_header_splits)

    if "--debug-prints" in args:
        for chunk in splits:
            print(len(chunk.page_content), repr(chunk.page_content[:100]))

    # Initialize vectorDB and add docs
    if os.path.exists("chroma_db"):
        if "--force" in args:
            shutil.rmtree("chroma_db")
        else:
            print(
                "Error: chroma_db already exists. Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    embedding = GPT4AllEmbeddings(
        model_name="nomic-embed-text-v1.5.f16.gguf", gpt4all_kwargs={}
    )  # type: ignore
    vectorstore = Chroma.from_documents(
        documents=splits,
        collection_name="cully-grove-bylaws",
        embedding=embedding,
        persist_directory="./chroma_db",
    )

    if "--debug-prints" in args:
        print(f"{vectorstore._collection.count()=}")


if __name__ == "__main__":
    main()
