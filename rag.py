"""
Adapted from:

https://python.langchain.com/docs/tutorials/rag/
"""

import argparse

from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


def main(args):
    load_environment()

    print("Loading data...")
    if args.path is not None:
        from custom import load_custom_docs
        docs = load_custom_docs(path=args.path, exclude="**/*.png",
                                sample_size=args.limit)
    else:
        docs = get_context_documents()

    print(f"Loaded {len(docs)} docs.")

    llm, embeddings = get_llm_embeddings()
    vector_store = get_vector_store(embeddings)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": args.q})
    print(response["answer"])


def get_context_documents():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs


def get_llm_embeddings():
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model="open-mistral-nemo")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return llm, embeddings


def get_vector_store(embeddings):
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)
    return vector_store


def build_parser():
    parser = argparse.ArgumentParser(description="Ask this RAG a question")
    parser.add_argument(
        "--q",
        required=True,
        help="Question"
    )
    parser.add_argument(
        "--path",
        help="Use this local directory as context, instead of default data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit loaded files to this number (int)"
    )
    return parser


class State(TypedDict):
    """Define state for application"""
    question: str
    context: List[Document]
    answer: str


def load_environment():
    """Get LLM and Langchain keys

    MISTRAL_API_KEY=ABC
    LANGCHAIN_API_KEY=ABC

    see:
    https://python.langchain.com/docs/tutorials/rag/
    """
    load_dotenv()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    main(args)
