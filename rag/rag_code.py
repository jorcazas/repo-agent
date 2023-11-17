import os

# from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language

# Set env var OPENAI_API_KEY or load from a .en file
import dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# encapsulate this into a function: load_repo


repo_path = "/Users/localhost/Documents/deep_dive/aladdin/aladdin-repo"
dotenv.load_dotenv()


def load_repo(repo_path):
    """
    Load a repo from a path

    Args:
        repo_path (str): Path to repo

    Returns:
        documents (List[Document]): List of documents
    """
    # Clone

    # repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

    # Load
    loader = GenericLoader.from_filesystem(
        path=repo_path,
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents = loader.load()

    return documents


def build_retriever(documents):
    """
    Build a retriever from a list of documents
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    texts = python_splitter.split_documents(documents)
    print(len(texts))

    # if directory exists
    # if os.path.exists("./chroma_db"):
    #     db = Chroma.
    # else:
    db = Chroma.from_documents(
        texts,
        OpenAIEmbeddings(disallowed_special=()),
        persist_directory="./chroma_db",
    )
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    return retriever


def qa(retriever, questions):
    """
    Build a QA system from a retriever
    """
    llm = ChatOpenAI(model_name="gpt-4", max_tokens=1000)
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    for question in questions:
        result = qa(question)
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")


if __name__ == "__main__":
    questions = [
        "¿Qué hace la interfaz SW?",
        "¿Cómo podrías mejorar la interfaz SW sabiendo que la interfaz MO es similar pero tiene mejores prácticas?",
    ]
    documents = load_repo(repo_path)
    retriever = build_retriever(documents)
    qa(retriever, questions)
