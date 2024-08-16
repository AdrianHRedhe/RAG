import os
import re
import glob
from datetime import datetime

import torch
import numpy as np
from PyPDF2 import PdfReader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import LatexTextSplitter


def get_chunks(text, tokenizer, max_length=510, overlap=30):
    """
    Splits text into chunks of a specified maximum length with overlap.

    Args:
        text (str): The text to split.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        max_length (int, optional): The maximum length of each chunk in tokens. Defaults to 510.
        overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 30.

    Returns:
        List[str]: A list of text chunks.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))

        if end == len(tokens):
            start = end - max_length

        encoded_chunk = tokens[start:end]
        chunk = tokenizer.batch_decode(encoded_chunk)
        chunks.append(chunk)

        start += max_length - overlap  # Move the starting point with overlap

    return chunks


def load_and_chunk(glob_path_to_docs="PDFs/*.pdf"):
    chunks = []
    chunk_paths = []
    file_paths = glob.glob(glob_path_to_docs)

    text_splitter = LatexTextSplitter(chunk_size=500, chunk_overlap=50)

    for it, file_path in enumerate(file_paths):
        if (it % 11 == 0) or (it == len(file_paths)):
            print(f"{round(it/(len(file_paths)) * 100,1)} % \t| {datetime.now()}")

        reader = PdfReader(file_path)
        pages = reader.pages

        pdf_text = ""
        for page in pages:
            pdf_text += page.extract_text() + "\n"

        # remove invalid UTF-8 characters
        pdf_text = pdf_text.encode("utf-8", "ignore").decode("utf-8")

        # remove extra whitespace
        pdf_text = re.sub(r"\s+", " ", pdf_text)

        chunk = text_splitter.create_documents([pdf_text])
        # chunk = get_chunks(pdf_text, tokenizer)

        if isinstance(chunk, list):
            chunks += chunk
            chunk_paths += [file_path] * len(chunk)
        else:
            chunks.append(chunk)
            chunk_paths.append(file_path)

    # chunks = [chunk.page_content for chunk in chunks]
    for i, chunk in enumerate(chunks):
        chunks[i].metadata["source"] = chunk_paths[i]
        chunks[i].metadata["id"] = i

    return chunks


class MyHfEmbeddings(Embeddings):
    """A class to make it possible to use self-defined embedding functions and
    models rather than only those supported by langchain."""

    def __init__(
        self, embed_function, tokenizer, model, device, sentence_embeddings=False
    ):
        self.embed_function = embed_function  # Self-defined embedding function
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.sentence_embeddings = sentence_embeddings

    def embed_documents(self, texts):
        """Embed a list of documents."""
        if not self.sentence_embeddings:
            embeddings = [
                self.embed_function(text, self.tokenizer, self.model, self.device)
                for text in texts
            ]
            return np.array(embeddings)
        else:
            return self.embed_function(texts, self.model, self.device)

    def embed_query(self, text):
        """Embed a query."""
        embedding = self.embed_function(text, self.model, self.device)
        return embedding


def embedding_function_BERT(text, tokenizer, model, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.pooler_output.cpu().squeeze().numpy()


def embedding_function_MISTRAL(text, tokenizer, model, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.cpu().squeeze().numpy()


def embedding_function_Sentence(text, model, device="cpu"):
    return model.encode(text)


# ---- PDF opener functions -------------------------------------------------------------------------


def open_pdf_at_location(pdf_path, page_num, passage_to_find=None):
    if passage_to_find != None:
        evince_command = f"evince --page-index={page_num + 1} --find='{passage_to_find}' '{pdf_path}'"
        os.system(evince_command)
        return

    evince_command = f"evince --page-index={page_num + 1} '{pdf_path}'"
    os.system(evince_command)


def get_pdf_page_num(pdf_path, passage_to_find):
    pdf = PdfReader(pdf_path)
    pages = pdf.pages
    for page_num, page in enumerate(pages):
        page_text = page.extract_text()
        # Ensure there are only singular whitespaces.
        page_text = re.sub(r"\s+", " ", page_text)

        # If there is another page. Add the first 499 characters of
        # that page to capture the chunks that stretch between pages.
        if page_num < (len(pages) - 1):
            next_page_text = pages[page_num + 1].extract_text()
            next_page_text = re.sub(r"\s+", " ", next_page_text)
            page_text = page_text + " " + next_page_text[:499]

        # Search for the passage
        if passage_to_find in page_text:
            return page_num
    raise ValueError("Could not find the text in the PDF.")


def open_pdf_at_passage_if_found(page_num, pdf_path, passage_to_find):
    # Get fulltext pdf.
    pdf = PdfReader(pdf_path)
    pages = pdf.pages
    pdf_text = ""

    for page in pages:
        page_text = page.extract_text()
        pdf_text += page_text
    
    # Search for the passage
    if passage_to_find in pdf_text:
        open_pdf_at_location(pdf_path, page_num, passage_to_find)
        return

    for sentence in passage_to_find.split("."):
        if len(sentence) > 10:
            if sentence in pdf_text:
                open_pdf_at_location(pdf_path, page_num, sentence)
                return

            pdf_text2 = pdf_text.replace("\n", " ")
            if sentence in pdf_text2:
                open_pdf_at_location(pdf_path, page_num, sentence)
                return

    open_pdf_at_location(pdf_path, page_num)
    return
