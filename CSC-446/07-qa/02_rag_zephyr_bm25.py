from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


store = InMemoryDocumentStore()
store.write_documents([
    Document(content="The Catholic University of America (CUA) is a private Catholic research university in Washington, D.C."),
    Document(content="The Vatican City is the smallest independent state in the world."),
    Document(content="CUA's campus is located in the Brookland neighborhood, often called Little Rome."),
])

retriever = InMemoryBM25Retriever(document_store=store, top_k=1)
prompt = PromptBuilder(
    template=(
        "You are a helpful assistant. Use ONLY the provided context. "
        "Answer the single question concisely in one sentence.\n\n"
        "Context:\n"
        "{% for d in context %}- {{ d.content }}\n{% endfor %}\n"
        "Question: {{ question }}\n"
        "Answer:"
    ),
    required_variables={"context", "question"},
)
llm = HuggingFaceLocalGenerator(
    model="HuggingFaceH4/zephyr-7b-beta",
    generation_kwargs={"max_new_tokens": 32, "do_sample": False},
)

pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("prompt", prompt)
pipe.add_component("llm", llm)
pipe.connect("retriever.documents", "prompt.context")
pipe.connect("prompt", "llm.prompt")


def main():
    question = "What can you tell me about the Vatican?"#"Where is CUA located?"
    result = pipe.run({
        "retriever": {"query": question},
        "prompt": {"question": question},
    })
    print(result["llm"]["replies"][0].strip())


if __name__ == "__main__":
    main()
