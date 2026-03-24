from haystack import Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


store = InMemoryDocumentStore()
store.write_documents([
    Document(content="The Catholic University of America (CUA) is in Washington, D.C."),
    Document(content="The Vatican City is enclaved within Rome, Italy."),
    Document(content="Rome is the capital city of Italy."),
])


def main():
    retriever = InMemoryBM25Retriever(document_store=store)
    question = "Where is the Vatican located?"
    docs = retriever.run(query=question)["documents"]
    context = "\n\n".join(d.content for d in docs)

    template = (
        "You are a precise QA assistant. Use ONLY the provided context.\n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{question}}\n"
        "Answer concisely with evidence. If unknown, say 'Not in context.'"
    )
    prompt_builder = PromptBuilder(template=template, required_variables={"context", "question"})
    prompt = prompt_builder.run(context=context, question=question)["prompt"]

    generator = HuggingFaceLocalGenerator(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        huggingface_pipeline_kwargs={"device_map": "auto", "dtype": "auto"},
        generation_kwargs={"max_new_tokens": 160, "do_sample": False},
    )
    generator.warm_up()
    answer = generator.run(prompt=prompt)["replies"][0]
    print(answer)


if __name__ == "__main__":
    main()
