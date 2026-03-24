import time

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


store = InMemoryDocumentStore()
store.write_documents([
    Document(content="CUA is in Washington, D.C."),
    Document(content="The Vatican City is enclaved within Rome, Italy."),
    Document(content="Rome is the capital city of Italy."),
])

retriever = InMemoryBM25Retriever(document_store=store)
template = (
    "You are a precise QA assistant. Use ONLY the provided context.\n\n"
    "Context:\n{% for doc in documents %}- {{ doc.content }}\n{% endfor %}\n\n"
    "Question: {{ question }}\n"
    "Answer concisely with evidence. If unknown, say 'Not in context.'"
)
question = "Where is CUA located?"


def build_pipeline(model_id, task):
    prompt_builder = PromptBuilder(template=template, required_variables={"question", "documents"})
    generator = HuggingFaceLocalGenerator(
        model=model_id,
        task=task,
        huggingface_pipeline_kwargs={"device_map": "auto", "dtype": "auto"},
        generation_kwargs={"max_new_tokens": 160, "do_sample": False},
    )
    generator.warm_up()
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "generator.prompt")
    return pipe


def run_pipeline(label, model_id, task):
    pipe = build_pipeline(model_id, task)
    start = time.perf_counter()
    out = pipe.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})
    latency = time.perf_counter() - start
    print(f"\n[{label}]\n")
    print(out["generator"]["replies"][0].strip())
    print(f"Latency: {latency:.2f}s")


def main():
    run_pipeline("Modern / Mistral", "mistralai/Mistral-7B-Instruct-v0.3", "text-generation")
    run_pipeline("Legacy / FLAN-T5", "google/flan-t5-base", "text2text-generation")


if __name__ == "__main__":
    main()
