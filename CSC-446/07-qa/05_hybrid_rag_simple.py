from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


store = InMemoryDocumentStore()
store.write_documents([
    Document(content="CUA is in Washington DC."),
    Document(content="The Vatican is in Rome, Italy."),
])


def main():
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    text_embedder.warm_up()
    embedded_docs = doc_embedder.run(documents=store.filter_documents())["documents"]
    store.write_documents(embedded_docs, policy="overwrite")

    bm25 = InMemoryBM25Retriever(document_store=store, top_k=3)
    dense = InMemoryEmbeddingRetriever(document_store=store, top_k=3)

    pipe = Pipeline()
    pipe.add_component("bm25", bm25)
    pipe.add_component("dense", dense)
    pipe.add_component("text_embedder", text_embedder)
    pipe.connect("text_embedder.embedding", "dense.query_embedding")

    question = "Where is CUA located?"
    results = pipe.run({"bm25": {"query": question}, "text_embedder": {"text": question}})
    docs = results["bm25"]["documents"] + results["dense"]["documents"]
    context = "\n".join({d.content for d in docs})

    prompt = PromptBuilder(
        template=(
            "Use ONLY the context below to answer the question in one short sentence.\n\n"
            "Context:\n{{context}}\n\n"
            "Question: {{question}}\n"
            "Answer:"
        ),
        required_variables={"context", "question"},
    )
    llm = HuggingFaceLocalGenerator(
        model="HuggingFaceH4/zephyr-7b-beta",
        generation_kwargs={"max_new_tokens": 24, "do_sample": False, "return_full_text": False},
    )
    llm.warm_up()
    final_prompt = prompt.run(context=context, question=question)["prompt"]
    answer = llm.run(prompt=final_prompt)["replies"][0]
    print(answer.strip())


if __name__ == "__main__":
    main()
