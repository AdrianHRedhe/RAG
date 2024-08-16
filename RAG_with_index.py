# %% IMPORTS
import os
import dotenv

from ragatouille import RAGPretrainedModel
from transformers import AutoTokenizer
from huggingface_hub import login
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

from RAG_helper_functions import *


# %% 0. PARAMETERS
k_retriever = 30
k_rerank = 10

vector_db_path = "faiss_vector_db"

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
rag_reranker_model_name = "colbert-ir/colbertv2.0"
llm_model_name = "mistral-large-latest"


# %% 1. DEFINE AND CHOOSE QUERY
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


task = "Given a web search query, retrieve relevant passages that answer the query"
queries = [
    get_detailed_instruct(task, "Which paper talks about cross-view geolocalization"),
    get_detailed_instruct(task, "What is the size of the cosplace embeddings?"),
    get_detailed_instruct(
        task, "Which paper talks about 100 meter distance thresholds?"
    ),
]
query = queries[1]


# %% 2. Load .env variables and log in to hf.
dotenv.load_dotenv(".env")
hf_token = os.environ["hftoken"]
login(hf_token)


# %% 3. Load Model
model = SentenceTransformer(embedding_model_name)
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
device = "cpu"
model.to(device)

langchain_embedding_model = MyHfEmbeddings(
    embed_function=embedding_function_Sentence,
    tokenizer=tokenizer,
    model=model,
    device=device,
    sentence_embeddings=True,
)


# %% 4. Load Vector DB
retriever = FAISS.load_local(
    vector_db_path,
    langchain_embedding_model,
    allow_dangerous_deserialization=True,
).as_retriever(search_kwargs={"k": k_retriever})


# %% 5. Create ReRankRetriever
RAG = RAGPretrainedModel.from_pretrained(rag_reranker_model_name)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=RAG.as_langchain_document_compressor(k=k_rerank),
    base_retriever=retriever,
)


# %% 6. CREATE THE FULL LLM PROMPT
def get_full_llm_prompt(query, contexts):
    prompt = """Instruct: Given a context of retrieved passages, try to answer the query. \
        If you do not know the answer, say so rather than take a guess. Finally, please also \
            provide an answer to which of the documents led you to the conclusion based on the \
                document paths."""

    for nr, context in enumerate(contexts):
        prompt += f"\nContext {nr}:\n\t text: {context.page_content}\n\t document_path: {context.metadata['source']}"

    prompt += f'\nQuery: {query.split("Query: ")[1]}\nAnswer:'
    return prompt


contexts = compression_retriever.invoke(query)
prompt = get_full_llm_prompt(query, contexts)
# You can paste this prompt into any LLM such as chatGPT.
print(prompt)


# %% 7. Recieve the LLM answer (OPTION 1: Mistral API).
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)

messages = [ChatMessage(role="user", content=prompt)]

chat_response = client.chat(
    model=llm_model_name,
    messages=messages,
)

print(chat_response.choices[0].message.content)


# %% 8. OPTIONAL take a look at the context described by Mistral.
# Write the context number to try to find it.
context_nr = 1

context = contexts[context_nr]
pdf_path = context.metadata["source"]
passage_to_find = context.page_content

page_num = get_pdf_page_num(pdf_path, passage_to_find)

print(f'The passage you are looking for is:\n{passage_to_find}')

open_pdf_at_passage_if_found(page_num, pdf_path, passage_to_find)

# %%
