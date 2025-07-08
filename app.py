import sys
import chromadb
import requests
from process_pdf import extract_text
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 0. Check if Ollama is running ---
def check_ollama_server():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434")
        response.raise_for_status()
        print("Ollama server is running.")
        return True
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
        print("Error: Ollama server not found.")
        print("Please make sure the Ollama application is running before executing this script.")
        return False

if not check_ollama_server():
    sys.exit(1) # Exit the script if the server isn't running

# --- 1. Configure Global Settings ---
# This setup uses a local LLM and a local embedding model.
print("Configuring global settings...")
Settings.llm = Ollama(model="tinyllama", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.chunk_size = 512 # Set the chunk size for splitting documents

# --- 2. Load Documents ---
print("Loading documents...")
input_pdf_path = "./documents/CCC2E.pdf"
plain_txt_path = extract_text(input_pdf_path, "./documents/CCC2E-ForRag.txt")

print(f"Loading documents from {plain_txt_path}...")
documents = SimpleDirectoryReader(input_files=[plain_txt_path]).load_data()

# --- 3. Setup the Vector Database (ChromaDB) ---
print("Setting up vector database...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 4. Create the Index ---
# This will process your documents, embed them, and store them in the database.
# This step can take some time depending on the number of documents.
print("Creating index... (This may take a while for large document sets)")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
print("Index created successfully.")

# --- 5. Setup the Query Engine ---
# This engine is what will answer our questions.
print("Setting up query engine...")
# This custom prompt ensures the model ONLY uses the provided documents.
qa_prompt_template_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_template = PromptTemplate(qa_prompt_template_str)

query_engine = index.as_query_engine(
    similarity_top_k=5, # Retrieve top 3 most similar documents
    text_qa_template=qa_prompt_template
)
print("Query engine is ready.")

# --- 6. Start Querying ---
# Now you can ask questions in a loop.
while True:
    query = input("\nEnter your query (or type 'thank you, bye' to quit): ")
    if query.lower() == 'thank you, bye':
        break
    if query.strip() == "":
        continue

    print("Querying the engine...")
    response = query_engine.query(query)
    print("\nResponse:")
    print(str(response))
