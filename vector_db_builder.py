"""
Vector Database Builder
Builds a vector database from recipe CSV data using Chroma and HuggingFace embeddings
"""

import os
from dotenv import load_dotenv
import pandas as pd
import time
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Login to HuggingFace
token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    login(token=token)
else:
    raise ValueError("HuggingFace token not found in environment variables")


class RecipeVectorDBBuilder:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory="./recipe_chroma_db"):
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.embedding = None
        
    def initialize_embedding_model(self):
        """Initialize the embedding model"""
        print(f"Initializing embedding model ({self.embedding_model})...")
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        print("Embedding model initialized.\n")
        
    def load_csv_data(self, csv_path):
        """Load and validate CSV data"""
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        print(f"Loading CSV from {csv_path}...")
        try:
            df = pd.read_csv(csv_path).fillna("")
            print(f"Loaded {len(df)} rows from CSV.\n")
            
            # Validate required columns
            required_columns = ['recipe_title', 'description', 'region', 'subregion', 
                              'ingredients', 'servings', 'prep_time', 'cook_time', 'steps']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns in CSV: {missing_columns}")
                # Add missing columns with empty strings
                for col in missing_columns:
                    df[col] = ""
            
            return df
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise
    
    def convert_to_documents(self, df):
        """Convert DataFrame rows to LangChain Documents"""
        documents = []
        print("Converting rows to LangChain Documents...")
        
        for idx, row in df.iterrows():
            content = f"""Recipe: {row.get('recipe_title', '')}
Description: {row.get('description', '')}
Region: {row.get('region', '')}
Subregion: {row.get('subregion', '')}
Ingredients: {row.get('ingredients', '')}
Servings: {row.get('servings', '')}
Prep Time: {row.get('prep_time', '')}
Cook Time: {row.get('cook_time', '')}
Steps: {row.get('steps', '')}"""

            metadata = {
                "index": idx,
                "title": row.get("recipe_title", ""),
                "region": row.get("region", ""),
                "subregion": row.get("subregion", ""),
                "author": row.get("author", ""),
                "vegan": row.get("vegan", ""),
            }

            documents.append(Document(page_content=content, metadata=metadata))
            
            if idx % 5000 == 0 and idx > 0:
                print(f" - Processed {idx} rows...")
        
        print(f"Converted {len(documents)} rows into Document objects.\n")
        return documents
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=100):
        """Split documents into smaller chunks"""
        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.\n")
        return chunks
    
    def create_vector_db(self, chunks):
        """Create and persist vector database"""
        if not self.embedding:
            self.initialize_embedding_model()
            
        print("Creating vector database and embedding documents (this may take a while)...")
        start_time = time.time()
        
        db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        db.persist()
        
        end_time = time.time()
        print(f"Vector DB created and persisted at '{self.persist_directory}'.")
        print(f"Total embedding and vector DB creation time: {(end_time - start_time)/60:.2f} minutes.\n")
        
        return db
    
    def build_from_csv(self, csv_path, chunk_size=1000, chunk_overlap=100):
        """Complete pipeline: CSV to Vector DB"""
        try:
            # Load CSV data
            df = self.load_csv_data(csv_path)
            
            # Convert to documents
            documents = self.convert_to_documents(df)
            
            # Split documents
            chunks = self.split_documents(documents, chunk_size, chunk_overlap)
            
            # Create vector DB
            db = self.create_vector_db(chunks)
            
            return db
            
        except Exception as e:
            print(f"Error building vector database: {e}")
            raise
    
    def load_existing_db(self):
        """Load existing vector database"""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"Vector database not found at: {self.persist_directory}")
        
        if not self.embedding:
            self.initialize_embedding_model()
            
        print(f"Loading existing vector database from {self.persist_directory}...")
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        print("Vector database loaded successfully!")
        return db


def main():
    """Example usage"""
    builder = RecipeVectorDBBuilder()
    
    # Example CSV path - update this to your actual CSV file path
    csv_path = "recipes.csv"  # Update this path
    
    try:
        # Build vector database from CSV
        db = builder.build_from_csv(csv_path)
        
        # Test the database
        retriever = db.as_retriever(search_kwargs={"k": 3})
        test_results = retriever.get_relevant_documents("Italian pasta recipe")
        
        print(f"Test query returned {len(test_results)} results")
        for i, doc in enumerate(test_results):
            print(f"Result {i+1}: {doc.metadata.get('title', 'No title')}")
        
    except FileNotFoundError:
        print("Please update the csv_path variable with the correct path to your recipe CSV file")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()