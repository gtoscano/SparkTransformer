# 🎓 ClassAdvisor

ClassAdvisor is a simple demo app that shows how easy it is to build an **AI-powered course suggestion system** using  
[Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and [OpenAI](https://platform.openai.com/).

This project is used in the **NLP class at The Catholic University of America** to illustrate how we can combine
natural language interfaces with real datasets (in this case, our Spring 2025 CS course catalog).

---

## 🚀 Quickstart (with Pipenv)

We use [Pipenv](https://pipenv.pypa.io/en/latest/) for dependency management.

```bash
# clone the repo (or download the code files)
cd classadvisor

# enter the pipenv shell
pipenv shell

# install dependencies
pipenv install
```

---

## 🔑 Set your API key

You’ll need an [OpenAI API key](https://platform.openai.com/account/api-keys).

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

---

## ▶️ Run the app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## ✨ Features

- Uses **Spring 2025 CS classes** as the catalog (auto-loaded).
- Semantic search with **OpenAI embeddings + FAISS**.
- LLM-generated suggestions with reasoning:

  - Matches your **interests** and **completed courses**.
  - Respects **credit limits, course level, and schedule blockers**.
  - Flags prerequisites and workload notes.

- Built with only a few dozen lines of Python + Streamlit.

---

## 📚 Learning Goals

- Show how natural language can drive structured outputs (recommendations in JSON).
- Explore how embeddings + retrieval can ground an LLM in real course data.
- Inspire you to prototype your own **AI apps** quickly!

---

## 🛠 Requirements

Dependencies are listed in `Pipfile`, but the key packages are:

- `streamlit`
- `pandas`
- `langchain`
- `langchain-openai`
- `faiss-cpu`

---

Happy coding, and welcome to **NLP @ CUA**! 🦅

