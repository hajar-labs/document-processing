from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re
from typing import List, Optional
import os
from pathlib import Path

class MTLChatSystem:
    def __init__(self, persist_directory: str = "./mtl_vectordb"):
        """
        Initialize the MTL Chat System with enhanced configuration

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self.setup_logging()

        # Configuration des embeddings - Using a more suitable French model
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/distiluse-base-multilingual-cased",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            # Fallback to a simpler model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # Initialize or load vector store
        self.vectorstore = self._initialize_vectorstore()

        # Configuration du modèle de langage avec des paramètres optimisés
        self.llm = self._initialize_llm()

        # Custom prompt template for MTL context
        self.prompt_template = self._create_prompt_template()

        # Chaîne RAG avec prompt personnalisé
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.5
                }
            ),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mtl_chatbot.log'),
                logging.StreamHandler()
            ]
        )

    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        if os.path.exists(self.persist_directory):
            logging.info("Loading existing vector store...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logging.info("Creating new vector store...")
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return vectorstore

    def _initialize_llm(self):
        """Initialize the language model with optimized parameters"""
        try:
            # Using a more suitable model for French and conversation
            llm = HuggingFacePipeline.from_model_id(
                model_id="microsoft/DialoGPT-medium",
                task="text-generation",
                model_kwargs={
                    "temperature": 0.3,
                    "max_length": 512,
                    "do_sample": True,
                    "pad_token_id": 50256,
                    "repetition_penalty": 1.1
                }
            )
            return llm
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            raise

    def _create_prompt_template(self):
        """Create a custom prompt template for MTL context"""
        template = """Tu es un assistant officiel du Ministère du Transport et de la Logistique du Maroc.

Utilise uniquement les documents suivants pour répondre à la question. Si l'information n'est pas disponible dans les documents, indique-le clairement.

Documents de référence:
{context}

Question: {question}

Instructions:
- Réponds uniquement en français
- Base ta réponse sur les documents fournis
- Sois précis et professionnel
- Si tu ne trouves pas l'information, dis "Je ne trouve pas cette information dans les documents disponibles"
- Cite les sources quand c'est pertinent

Réponse:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None):
        """
        Add documents to the vector store

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            doc_objects = []
            for i, doc in enumerate(documents):
                chunks = text_splitter.split_text(doc)
                for chunk in chunks:
                    metadata = metadatas[i] if metadatas else {}
                    doc_objects.append(Document(page_content=chunk, metadata=metadata))

            # Add to vector store
            self.vectorstore.add_documents(doc_objects)
            self.vectorstore.persist()

            logging.info(f"Added {len(doc_objects)} document chunks to vector store")

        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            raise

    def load_documents_from_directory(self, directory_path: str):
        """
        Load all text documents from a directory

        Args:
            directory_path: Path to directory containing documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            logging.error(f"Directory {directory_path} does not exist")
            return

        documents = []
        metadatas = []

        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": str(file_path), "filename": file_path.name})

            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")

        if documents:
            self.add_documents(documents, metadatas)
            logging.info(f"Loaded {len(documents)} documents from {directory_path}")

    def answer_question(self, question: str) -> dict:
        """
        Answer a question using the RAG system

        Args:
            question: User question

        Returns:
            Dictionary with answer, sources, and confidence info
        """
        try:
            # Enhanced question with context
            enhanced_question = f"""
            Question concernant les services du Ministère du Transport et de la Logistique du Maroc:
            {question}
            """

            # Get response with sources
            result = self.qa_chain({"query": enhanced_question})

            # Post-process the response
            processed_answer = self.post_process_response(result["result"])

            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content_preview": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)

            return {
                "answer": processed_answer,
                "sources": sources,
                "original_question": question,
                "status": "success"
            }

        except Exception as e:
            logging.error(f"Error answering question: {e}")
            return {
                "answer": "Désolé, une erreur s'est produite lors du traitement de votre question.",
                "sources": [],
                "original_question": question,
                "status": "error",
                "error": str(e)
            }

    def post_process_response(self, response: str) -> str:
        """
        Clean and format the model response

        Args:
            response: Raw response from the model

        Returns:
            Cleaned response
        """
        # Remove repetitive patterns
        response = re.sub(r'(.{20,}?)\1+', r'\1', response)

        # Clean up extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'

        # Ensure professional tone
        if not response.endswith(('.', '!', '?')):
            response += '.'

        return response

    def search_similar_documents(self, query: str, k: int = 5) -> List[dict]:
        """
        Search for similar documents without generating an answer

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents with scores
        """
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            results = []

            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })

            return results

        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []

    def get_vectorstore_info(self) -> dict:
        """Get information about the vector store"""
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "distiluse-base-multilingual-cased",
                "status": "active"
            }
        except Exception as e:
            logging.error(f"Error getting vectorstore info: {e}")
            return {"status": "error", "error": str(e)}