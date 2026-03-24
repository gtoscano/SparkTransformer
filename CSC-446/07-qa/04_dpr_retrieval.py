from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


store = InMemoryDocumentStore()
store.write_documents([
    Document(content="The Vatican City is enclaved within Rome, Italy."),
    Document(content="CUA is located in Washington, D.C."),
])


def main():
    doc_embedder = SentenceTransformersDocumentEmbedder(model="facebook-dpr-ctx_encoder-single-nq-base")
    doc_embedder.warm_up()
    docs = list(store.filter_documents())
    embedded_docs = doc_embedder.run(documents=docs)["documents"]
    store.write_documents(embedded_docs, policy=DuplicatePolicy.OVERWRITE)

    retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=2)
    query_embedder = SentenceTransformersTextEmbedder(model="facebook-dpr-question_encoder-single-nq-base")
    query_embedder.warm_up()
    query_embedding = query_embedder.run(text="Where is the Vatican located?")["embedding"]
    results = retriever.run(query_embedding=query_embedding)

    for doc in results["documents"]:
        print(doc.content)


if __name__ == "__main__":
    main()
