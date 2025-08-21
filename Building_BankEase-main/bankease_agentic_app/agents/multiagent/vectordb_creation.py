'''
To create the vector db used by RAGBot.
Its a one-time activity used only when there are new data to be ingested.
Add the files to be ingested into file_list.
Using RecursiveCharacterTextSplitter to chunk text and FAISS Vector db.
Vector db -> faiss_database in trunk 
'''

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))      # Move up one level to reach trunk
DATA_FILE1 = os.path.join(TRUNK_DIR, 'data', 'Dataset.txt')
DATA_FILE2 = os.path.join(TRUNK_DIR, 'data', 'policies.txt')
<<<<<<< HEAD

=======
# Define the list of files to be ingested; add the file path above and append to this list
>>>>>>> multiagent-sql-auth
file_list = [DATA_FILE1, DATA_FILE2]

documents = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=25)

for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = text_splitter.split_text(content)

    for chunk in chunks:
        doc = Document(page_content=chunk, metadata={"source": os.path.basename(file_path)})
        documents.append(doc)

vector_store = FAISS.from_documents(documents, embedding_function)

SAVE_DIR = os.path.join(TRUNK_DIR, 'faiss_database')

vector_store.save_local(SAVE_DIR)