# Simple RAG project
This is a very simple RAG project. It can help you generate prompts for answering questions about you personal PDF files.
It does so by vectorizing chunks of text from the PDF to embeddings which can be used to calculate the similarity between text.

The RAG or Retrieval Augmented Generation is basically conducted in three steps:
1. From a "database" of PDF files, you generate a Vector database of embeddings. This step is conducted "offline" and only needs to be done once.
2. When you have a question, or query for the documents, you simply write that question, turn that question into an embedding and calculates the similarity between the query and the "documents" in the vector database. From this you retrieve the most likely answers. These answers are then reranked using a different, more computationally expensive model.
3. Finally, using the original query, and the top-10 retrieved and re-ranked results from the PDFs you can generate a prompt, which an LLM can use to try to answer your question. See this [example](#example) to understand what it can look like.

## File structure:
```
.
├── README.md 				# You are here
├── RAG_helper_functions.py # Used in the other programs
├── RAG_create_index.py		# Gen a vector DB from PDFs
├── RAG_with_index.py		# Perform RAG with vector DB
├── environment.yml			# Conda environment
├── .gitignore				# Ensure no unwanted commits
├── .env					# Safekeep your tokens
├── PDFs
│   └── 					# Here you can put your PDFs
└── faiss_vector_db 		# Vector DB after running 
	├── index.faiss			  RAG_create_index.py
    └── index.pkl
```

## How to run:
Create a new conda environment from the environment.yml file.
```
conda env create -f environment.yml
```
Activate this new conda environment.
```
conda activate RAGatouille
```

* Put your HuggingFace token and optionally your Mistral token in your .env file.

* Put you PDF files in the [PDFs](PDFs) folder.

* Start running [RAG_create_index.py](RAG_create_index.py), and ensure that your interpreter is actually running the RAGatouille conda env.

* Once the program has finished running and you have your index 
you can start the [RAG_with_index_program](RAG_with_index_program). In step 0 you can switch parameters, such as which model is used. In step 1 you can define which query to ask. Run the program until step 6 to generate the prompt. This prompt can either be used as input for any LLM such as GPT, or be used in step 7.

* If you have gotten a Mistral token and put it in the [.env](.env) file, you can recieve the answer to your question directly in your interpreter in step 7.

* Step 8, can showcase where the answer is in the PDF. To run step 8 in the [RAG_with_index.py](RAG_with_index.py) file, you need to have evince on your computer. To download evince on mac simply run the following in the commandline:
```
brew install evince
```

## Example
This is a real example of a query asked to a database of PDFs on image retrieval and geolocalization, where several papers talk about the CosPlace model.

Query:
```
What is the size of the cosplace embeddings?
```

RAG Prompt generated:
```
Instruct: Given a context of retrieved passages, try to answer the query.
If you do not know the answer, say so rather than take a guess. 
Finally, please also provide an answer to which of the documents led you to the conclusion based on the document paths.
Context 0:
	 text: from D&C, CosPlace achieves a new SOTA (+ Figure 5. Example of our mixed pipeline . Thanks to the reduc- tion in the search space obtained via the predictions of D&C, the retrieval module correctly localizes the query Figure 6. Ablation on the values of MandN.Mdetermines cell size,Nis the distance between cells in a group. 6% LR@1), while being 500 times faster than the retrieval only version. Similarly, the NetVLAD model can achieve a speedup by 4 orders of magnitude and an increase in LR@1 by
	 document_path: PDFs/Trivigno et al. - 2023 - Divide&Classify Fine-Grained Classification for C.pdf
Context 1:
	 text: tex- tual queries. Retrieval is performed by ranking text-image similarity using cosine distance in a shared embeddings space, with potential acceleration using Large Scale Index (details in Section 3.4). This approach accommodates any dual-encoder architecture. We present top retrieved images for the query ’bird’ using Cluster-CLIP features. searching in application specific datasets (e.g., e-commerce, automotive, medical applica- tions). In both cases, scalability and efficiency play critical
	 document_path: PDFs/Levi et al. - 2023 - Object-Centric Open-Vocabulary Image-Retrieval wit.pdf
Context 2:
	 text: is perhaps one of the most important fac- tors when choosing an algorithm in the real world (given that for efﬁcient retrieval, all database descriptors should be kept in memory), we produce much more compact vectors w.r.t. previous methods. For instance, while [1, 27] use the full NetVLAD dimension of 32k, and [18, 32] reduce it to 4k, CosPlace achieves SOTA results with a dimension of just 512. Moreover, in Sec. 5.4 and Appendix B.3.1, we in- vestigate how results depend on the dimensionality
	 document_path: PDFs/Berton et al. - 2022 - Rethinking Visual Geo-localization for Large-Scale.pdf
Context 3:
	 text: concern. To help ameliorate this problem, the in- verted index is stored in a space-efﬁcient binary-packed structure. Additionally, when main memory is exhausted, the engine can be switched to use an inverted index ﬂat- tened to disk, which caches the data for the most frequently requested words. For example, for a vocabulary size of 1M words, our search engine implementation can query the combined 5K+100K datasets in approximately 0.1s for a typical query and the inverted index consumes 1GB of
	 document_path: PDFs/Philbin et al. - 2007 - Object retrieval with large vocabularies and fast .pdf
Context 4:
	 text: the corpus size, but in practice it is close to linear in the number of documents that match a given query, generally a major saving. For sparse queries, this can result in a substantial speedup, as only documents which contain words present in the query need to be examined. The scores for each docu- ment are accumulated so that they are identical to explicitly computing the similarity. With large corpora of images, memory usage becomes a major concern. To help ameliorate this problem, the in-
	 document_path: PDFs/Philbin et al. - 2007 - Object retrieval with large vocabularies and fast .pdf
Context 5:
	 text: re- quired to perform the similarity search grows linearly with the database size, thus leading to potentially unacceptable delays for applications. Recently, CosPlace [3] has ad- dressed the training time scalability problem by using an al- ternative approach to contrastive learning, allowing to learn from large scale databases and achieving state of the art re- sults across many datasets. Yet, it uses retrieval for infer- ence, thus the scalability problem at test time still persists.
	 document_path: PDFs/Trivigno et al. - 2023 - Divide&Classify Fine-Grained Classification for C.pdf
Context 6:
	 text: embedding for CLIP) into a large-scale index. Qualitative retrieval examples of interest are presented in Figures S3 and S4. Figure S3 shows the top retrieval results for ’Helicopter’, ’Wall clock’, and ’Bulldozer’ text queries. Using Cluster-CLIP allows the retrieval of cluttered images with relatively small instances of the requested category. Figure S4 shows top retrieval results for ’Wa- ter Tower’, ’Globe’, ’Passport’, ’Earplugs’, ’Lemon’ and ’Chickpea’ text queries, in which Cluster-CLIP
	 document_path: PDFs/Levi et al. - 2023 - Object-Centric Open-Vocabulary Image-Retrieval wit.pdf
Context 7:
	 text: pairings across a batch actually occurred. To do this, CLIP learns a with high pointwise mutual information as well as the names of all Wikipedia articles above a certain search volume. Finally all WordNet synsets not already in the query list are added.multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similar- ity of the image and text embeddings of the Nreal pairs in the batch while minimizing the cosine similarity of the embeddings of
	 document_path: PDFs/Radford et al. - 2021 - Learning Transferable Visual Models From Natural L.pdf
Context 8:
	 text: In sec- tion 3.3 we describe the number of descriptors and words used for quantization, but in all cases the visual vocabulary is computed on the 5Kdataset. Search Engine. Our search engine uses the vector-space model [7] of information-retrieval. The query and each doc- ument in the corpus is represented as a sparse vector of term (visual word) occurrences and search proceeds by calculat- ing the similarity between the query vector and each doc- ument vector, using an L2distance. We use the
	 document_path: PDFs/Philbin et al. - 2007 - Object retrieval with large vocabularies and fast .pdf
Context 9:
	 text: descriptors as in [49] for a classic retrieval over the database. This allows for the model to be used also on other datasets from unseen geographical areas (see Tab. 3). 5. Experiments 5.1. Implementation details Architecture. CosPlace is architecture-agnostic, i.e. it can be applied on virtually any image-based model. For most experiments, we rely on a simple network made of a stan- dard CNN backbone followed by a GeM pooling and a fully connected layer with output dimension 512. Note that
	 document_path: PDFs/Berton et al. - 2022 - Rethinking Visual Geo-localization for Large-Scale.pdf
Query: What is the size of the cosplace embeddings?
Answer:
```

Mistral LLM answer based on prompt:
```
The size of the CosPlace embeddings is 512 dimensions.

This information is derived from Context 2:
- document_path: PDFs/Berton et al. - 2022 - Rethinking Visual Geo-localization for Large-Scale.pdf
```