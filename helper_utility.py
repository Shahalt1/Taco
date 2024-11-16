import os
import numpy as np
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pypdf

def load_documents_from_directory(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as file:
                documents.append({"id": filename, "text": file.read()})
    print(f"Documents loaded from directory")
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    print(f"Documents split into {len(chunks)} chunks")
    return chunks

def clean_texts(pdf_texts:str, substrings_to_remove:list[str] = [], delete_list:list[int] = []) -> list[str]:
    """
    Removes specified substrings from each text in pdf_texts.
    
    Parameters:
    - pdf_texts (list of str): List of text strings to clean.
    - substrings_to_remove (list of str): List of substrings to remove from each text.
    
    Returns:
    - list of str: Cleaned list of text strings.
    """
    cleaned_texts = []
    for text in pdf_texts:
        for substring in substrings_to_remove:
            text = text.replace(substring, "")
        cleaned_texts.append(text)
    
    # Sort delete_list in descending order and remove indices in one pass
    if delete_list:
        for index in sorted(delete_list, reverse=True):
            if 0 <= index < len(cleaned_texts):
                del cleaned_texts[index]
            
    return cleaned_texts

def extract_text_from_pdfs(pdf_path:str) -> list[str]:
    reader = pypdf.PdfReader(pdf_path)
    # Extract text from each page, strip leading/trailing whitespace, and remove empty texts
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]  # Filter out empty text

    return pdf_texts

def generate_embeddings(text):
    embeddings = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")['embedding']
    embeddings = [np.float16(i) for i in embeddings]
    return embeddings

def query_documents(question, collection, n_results=2):
    # Query the collection to retrieve relevant documents
    results = collection.query(query_texts=question, n_results=n_results)
    # Flatten the documents if needed (only if results["documents"] is a list of lists)
    chunks = [doc for sublist in results["documents"] for doc in sublist]

    print("======= Returning relevant chunks =========")
    
    # Loop through the results to print each document's details
    for idx, document in enumerate(results["documents"][0]):  # Assuming it's a list of lists
        doc_id = results["ids"][0][idx]  # Correctly access the first list in "ids"
        distance = results["distances"][0][idx]  # Correctly access the first list in "distances"
        print(f"\nDocument ID: {doc_id}\nDistance: {distance}\nFound chunk: '{document[:50]}...'\n")

    return chunks

def generate_response(question, relevant_chunks, model):
    context = "\n\n".join(relevant_chunks)
    prompt = f"Give short Answer to the question using the following context: {context}, question: {question}"
    response = model.generate_content(prompt)
    return response.text

def tokenizer(text, chunk_size, chunk_overlap=0, tokens_per_chunk=256):
    charector_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\n\n", ". ", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap = chunk_overlap,
        tokens_per_chunk = tokens_per_chunk
    )
    charector_split_text = charector_splitter.split_text("\n\n".join(text))
    token_split_texts = []
    for text in charector_split_text:
        token_split_texts += token_splitter.split_text(text)
        
    return token_split_texts

def augment_query_generated(query, model):
    prompt = f"You are AI educational tutor for class 10th students of Kerala State Board Syllabus. Provide short answers to the corresponding questions : {query}"
    response = model.generate_content(prompt)
    return response.text

def generate_multi_query(query, model):
    prompt = """
        You are an AI assistant for high school students in 10th grade. Your users are inquiring about a specific textbook for their syllabus.
        For the given question, propose up to five related questions to assist them in finding the information they need.
        Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
        Ensure each question is complete and directly related to the original inquiry. 
        List each question on a separate line without numbering.
    """
    response = model.generate_content(prompt + query)
    return response.text.split("\n")

def reranker(encoder, retrieved_documents, original_query):
    unique_documents = set()
    for documents in retrieved_documents:
        for document in documents:
            unique_documents.add(document)
    unique_documents = list(unique_documents)

    pairs = []
    for doc in unique_documents:
        pairs.append([original_query, doc])
        
    scores = encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:5]
    top_documents = [unique_documents[i] for i in top_indices]
    
    return top_documents



def plot_query_generated(collection, embedding_functions,original_query, joint_query, retrieved_embeddings ):
    # Retrieve dataset embeddings and convert to a 2D array
    dataset_embeddings = collection.get(include=['embeddings'])['embeddings']

    # Generate individual embeddings for queries and retrieved items, reshaped to ensure 2D structure
    retrieved_embedding = np.array(embedding_functions([retrieved_embeddings])[0]).reshape(1, -1)
    original_query_embedding = np.array(embedding_functions([original_query])[0]).reshape(1, -1)
    augmented_query_embedding = np.array(embedding_functions([joint_query])[0]).reshape(1, -1)

    # Combine all embeddings into one array for t-SNE to embed them in the same 2D space
    all_embeddings = np.vstack([dataset_embeddings, retrieved_embedding, original_query_embedding, augmented_query_embedding])

    # Fit t-SNE on the combined embeddings
    tsne = TSNE(n_components=2, random_state=0)
    projected_all_embeddings = tsne.fit_transform(all_embeddings)

    # Extract the projected embeddings
    projected_dataset_embeddings = projected_all_embeddings[:len(dataset_embeddings)]
    projected_retrieved_embedding = projected_all_embeddings[len(dataset_embeddings)]
    projected_original_query_embedding = projected_all_embeddings[len(dataset_embeddings) + 1]
    projected_augmented_query_embedding = projected_all_embeddings[len(dataset_embeddings) + 2]

    # Plot the results
    plt.figure()

    # Plot dataset embeddings
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
        label="Dataset embedding"
    )
    # Plot retrieved embedding
    plt.scatter(
        projected_retrieved_embedding[0],
        projected_retrieved_embedding[1],
        s=100,
        facecolors="none",
        edgecolors="g",
        label="Retrieved query embedding"
    )
    # Plot original query embedding
    plt.scatter(
        projected_original_query_embedding[0],
        projected_original_query_embedding[1],
        s=150,
        marker="X",
        color="r",
        label="Original query embedding"
    )
    # Plot augmented query embedding
    plt.scatter(
        projected_augmented_query_embedding[0],
        projected_augmented_query_embedding[1],
        s=150,
        marker="X",
        color="orange",
        label="Augmented query embedding"
    )
    plt.legend(loc='upper right')
    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"{original_query}")
    plt.axis("off")
    plt.show()  # Display the plot
