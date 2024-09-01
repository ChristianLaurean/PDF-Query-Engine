import logging

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import load_configurations, configure_logging, PATH_FILE

configure_logging()


# Load the PDF file
def load_pdf() -> PyPDFLoader:
    """
    Load a PDF file into a PyPDFLoader object.

    This function takes a filename as input and attempts to load the PDF file into a PyPDFLoader object.
    If the file cannot be loaded, an error message is logged and None is returned.


    Returns:
    PyPDFLoader: A PyPDFLoader object containing the loaded PDF file, or None if an error occurred.
    """
    try:
        loader = PyPDFLoader(PATH_FILE)
        # "/home/christianlaurean/Documentos/IA/proyectos_IA/qa_from_pdf_file/data/"
        return loader.load()
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        assert NameError


def inicialize_openai() -> ChatOpenAI | None:
    """
    Initialize a ChatOpenAI instance with the provided API key and model.

    This function attempts to create a ChatOpenAI instance using the provided API key and model.
    If the initialization is successful, the function returns the ChatOpenAI instance.
    If an error occurs during initialization, an error message is logged, and the function returns None.

    Parameters:
    None

    Returns:
    ChatOpenAI | None: A ChatOpenAI instance if initialization is successful, or None if an error occurs.
    """
    try:
        return ChatOpenAI(api_key=load_configurations(), model="gpt-3.5-turbo-0125")
    except Exception as e:
        logging.error(f"Error initializing OpenAI: {e}")
        return None


def inicialize_chroma(split) -> Chroma:
    """
    Initialize a Chroma vector database using the provided documents and OpenAI embeddings.

    This function takes a list of documents as input and initializes a Chroma vector database.
    The documents are embedded using the OpenAI embeddings with the provided API key.
    The resulting vector database is then converted to a retriever and returned.

    Parameters:
    split (List[Document]): A list of documents to be used for initializing the Chroma vector database.

    Returns:
    Chroma: A Chroma retriever initialized with the provided documents and OpenAI embeddings.
    """
    vector_db = Chroma.from_documents(
        documents=split, embedding=OpenAIEmbeddings(api_key=load_configurations())
    )
    return vector_db.as_retriever()


def rag(pdf_file):
    """
    Initialize a Retrieval-Augmented Generation (RAG) system using a Chroma vector database.

    This function takes a PyPDFLoader object containing a PDF file as input, splits the PDF into smaller
    documents using a RecursiveCharacterTextSplitter, initializes a Chroma vector database with the split
    documents, and returns a retriever for the vector database.

    Parameters:
    pdf_file (PyPDFLoader): A PyPDFLoader object containing a PDF file.

    Returns:
    Chroma: A Chroma retriever initialized with the split documents from the PDF file.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split = splitter.split_documents(pdf_file)
    retriver = inicialize_chroma(split)
    return retriver


def chain(llm, retriever):
    """
    Initialize a Retrieval-Augmented Generation (RAG) chain using a language model (LLM) and a retriever.

    This function creates a chain that utilizes a language model (LLM) and a retriever to perform
    question-answering tasks. The chain consists of a prompt template, a question-answering chain,
    and a retrieval-augmented generation chain. The prompt template is designed to guide the LLM
    in generating answers based on retrieved context. The question-answering chain processes the
    LLM's output and the prompt template to generate an answer. The retrieval-augmented generation
    chain combines the retriever and the question-answering chain to provide a complete RAG system.

    Parameters:
    llm (ChatOpenAI): A language model instance for generating answers.
    retriever (Chroma): A retriever instance for retrieving relevant context from a database.

    Returns:
    rag_chain (LangChain.Chain): A Retrieval-Augmented Generation (RAG) chain that can be used to answer questions.
    """
    template = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. and language spanish"
        "\n\n"
        "{context}"
    )
    promt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{input}"),
        ]
    )

    question_aswered = create_stuff_documents_chain(llm, promt)
    rag_chain = create_retrieval_chain(retriever, question_aswered)

    return rag_chain


def main():
    """
    Main function to initialize the RAG system, load a PDF file, create a retrieval-augmented generation chain,
    and perform a question-answering task.

    This function connects to OpenAI, loads a PDF file, initializes a retrieval-augmented generation (RAG) chain,
    and uses the chain to answer a question about the loaded PDF file. The function prints the answer to the console.

    """
    # Connect to OpenAI
    llm = inicialize_openai()

    # Load the PDF file
    pdf = load_pdf()

    # Initialize a retrieval-augmented generation (RAG) chain
    retriever = rag(pdf)

    # Create a retrieval-augmented generation chain
    rag_chain = chain(llm, retriever)

    # Perform a question-answering task using the RAG chain
    response = rag_chain.invoke({"input": "Dame un resumen"})

    # Print the answer to the console
    print(response["answer"])


if __name__ == "__main__":
    main()
