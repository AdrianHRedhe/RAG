# %%
import os
import dotenv

from transformers import AutoTokenizer
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

from RAG_helper_functions import *

# %% 0. PARAMETERS
glob_path_to_docs = "PDFs/*.pdf"
vector_db_save_path = "faiss_vector_db"

if os.path.exists(vector_db_save_path):
    vector_db_save_path = None
    raise ValueError(
        f"THIS WILL OVERWRITE EXISTING VECTOR DB.\nChoose new path. Prev path: {vector_db_save_path}"
    )

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

# %% 1. Load .env variables and log in to hf.
dotenv.load_dotenv(".env")
hf_token = os.environ["hftoken"]
login(hf_token)

# %% 2. Load Model
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

# %% 3. Create docs
docs = load_and_chunk()

# %% 4. Create Vector DB
VectorDB = FAISS.from_documents(docs, langchain_embedding_model)

# %% 5. SAVE Vector DB
VectorDB.save_local(vector_db_save_path)

# %%
