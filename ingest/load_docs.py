from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# chunk
documents = SimpleDirectoryReader("data").load_data()

# build index and save to chromadb
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="chroma_storage")
