from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
import numpy as np
from dotenv  import load_dotenv
load_dotenv()
from langsmith import traceable

class Custom_Rag():        
    def __init__(self, model):
        openai_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4-0125-preview"]
        groq_models = ["gemma-7b-it", "mixtral-8x7b-32768", "llama3-8b-8192"]
        nvidia_api_models = ["databricks/dbrx-instruct", "microsoft/phi-3-small-8k-instruct", "google/gemma-7b", "meta/llama3-70b-instruct"]

        if model in nvidia_api_models:
            self.model = ChatNVIDIA(model=model, temperature=0.2)
        elif model in groq_models:
            self.model = ChatGroq(model=model, temperature=0.2)
        elif model in openai_models:
            self.model = ChatOpenAI(model=model, temperature=0.2)
        else:
            raise ValueError(f"Model '{model}' is not recognized. Please choose from OpenAI, NVIDIA, or Groq models.")
        
        print(self.model)


            
    def load_file(self, file_name):
        loader = PyPDFLoader(file_name).load()    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300)
        pages = text_splitter.split_documents(loader)
        return pages

    def generate_queries(self, query):
        template = """You are an AI language model assistant. Your task is to generate Four 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_querie = (
            prompt_perspectives
            | self.model 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
            | (lambda x: [query] + x)
        )
        return generate_querie 

    def _get(self, a):
        dd = []
        for s in a:
            dd.extend(s)
        return dd

    # @traceable(name="Removing duplicates documents")
    def get_unique_documents(self, doc_list):
        seen_content = set()
        unique_documents = []
        
        for doc in doc_list:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_documents.append(doc)
        
        del seen_content
        
        return unique_documents

    def keyword_extractor(self, query):
        prompt = """
        You are an AI language model assistant. Your task is to help the user retrieve keywords from their query. 

        Please provide me with the keywords you would like to extract from your query. 

        Keywords: {keywords}
        """
        prompt_perspectives = ChatPromptTemplate.from_template(prompt)
        
        generate_querie = (
            prompt_perspectives 
            | self.model 
            | StrOutputParser() )
        return generate_querie

    @traceable(name="custom-rag")
    def main(self, Query):
        chunks = self.load_file("resume.pdf")
        db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        faiss_retriever = db.as_retriever(search_kwargs={'k': 10})

        Bm25_retriever = BM25Retriever.from_documents(chunks)
        Bm25_retriever.k = 10

        map_chain = self.generate_queries(Query) | faiss_retriever.map() | self._get | self.get_unique_documents
        key_chain = self.keyword_extractor(Query) | Bm25_retriever | self.get_unique_documents

        ensemble_retriever = EnsembleRetriever(
            retrievers=[map_chain, key_chain], weights=[0.5, 0.5]
        )

        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=4)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        final_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:"""

        final_prompt_perspectives = ChatPromptTemplate.from_template(final_prompt)

        llm_chain = ({"context": itemgetter("query") | compression_retriever,
                "question": itemgetter("query")}
                | 
                RunnableParallel({
                    "response":  final_prompt_perspectives | self.model | StrOutputParser() ,
                    "context": itemgetter("context")
                })
                )
        
        return llm_chain.invoke({"query": Query})