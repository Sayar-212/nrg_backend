"""
RAG Chain Implementation
Implements the Retrieval-Augmented Generation chain for recipe creation
"""

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pathlib import Path


class RecipeRAGChain:
    def __init__(self, 
                 model_name="llama3.2:1b",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory="./recipe_chroma_db",
                 temperature=0.2):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.llm = None
        self.embedding = None
        self.vector_store = None
        self.qa_chain = None
        
    def initialize_llm(self):
        """Initialize the ChatOllama model"""
        print(f"Initializing LLM model: {self.model_name}")
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)
        
    def initialize_embedding(self):
        """Initialize the embedding model"""
        print(f"Initializing embedding model: {self.embedding_model}")
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
    def load_vector_store(self):
        """Load the vector store"""
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"Vector database not found at: {self.persist_directory}")
        
        if not self.embedding:
            self.initialize_embedding()
            
        print(f"Loading vector store from: {self.persist_directory}")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        
    def create_prompt_template(self):
        """Create the prompt template for recipe generation"""
        template = """You are an experienced culinary innovator. Master of making new/novel recipes which are edible and not fully imaginitive.
Create new, novel, practical and delicious recipes using ONLY the provided ingredients, authentically adapted to the specified regional cuisine style. Nothing else may be added beyond the given ingredients.

Use the following context from the recipe database to inform your response:
{context}

Based on the user's query, identify the region and ingredients, then create a new, novel authentic regional recipe.

User Query: {question}

FORMAT (follow exactly):

Recipe Title:
The title should be a summary of the dish in max 3 words, also reflecting the region/culture
Example: if query is India and ingredients are Chicken, Rice, Garam masala
Recipe title: Bhuna Murg Chawal

Taste: The taste of the dish should adhere to the preparation. If no sweetenery ingredients is added it should not predict sweet

Regional Context:
Brief explanation of how this dish authentically represents the regional cuisine and why these ingredients harmonize within this culinary tradition.

Ingredient List (for 4 servings):
Precise measurements treating each ingredient as essential within the regional style. No substitutions. No extras beyond what's listed.

Preparation Steps:
- Use clear numbered steps following traditional regional cooking sequences
- Include region-specific techniques and timing
- Guide the cook like a cultural mentor with sensory cues specific to the cuisine
- Reference traditional cooking wisdom and regional preferences
- Build narrative tension following the cuisine's typical preparation flow

Chef's Notes:
Reflect on the regional authenticity - how traditional techniques, cultural wisdom, and ingredient character unite to create something that honors the cuisine while being genuinely novel.

CONSTRAINTS:
- The recipe should be new, unorthodox and never seen before type
- Stay authentic to the regional cuisine's core principles and techniques and make new combinations.
- Create something new yet culturally grounded
- Focus on emotional satisfaction and cultural resonance
- The preparation should be stunning. Absolute like a human.

Recipe:"""
        
        return PromptTemplate.from_template(template)
        
    def setup_retriever(self, search_type="similarity_score_threshold", k=3, score_threshold=0.6):
        """Setup the retriever with specified parameters"""
        if not self.vector_store:
            self.load_vector_store()
            
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
            },
        )
        return retriever
        
    def create_qa_chain(self):
        """Create the QA chain"""
        if not self.llm:
            self.initialize_llm()
            
        prompt = self.create_prompt_template()
        retriever = self.setup_retriever()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
    def generate_recipe(self, region, ingredients):
        """Generate a recipe based on region and ingredients"""
        if not self.qa_chain:
            self.create_qa_chain()
            
        formatted_query = f"New recipe from Region: {region}, and using these Ingredients: {ingredients}"
        
        try:
            result = self.qa_chain({"query": formatted_query})
            return {
                "answer": result["result"],
                "context": result["source_documents"]
            }
        except Exception as e:
            print(f"Error generating recipe: {e}")
            return {
                "answer": f"Error generating recipe: {str(e)}",
                "context": []
            }
    
    def query(self, query_string):
        """
        Query the RAG chain with a user input string
        Expected format: "Region: ingredients" or just "ingredients" (defaults to Indian)
        """
        if ":" in query_string:
            region, ingredients = query_string.split(":", 1)
            region = region.strip()
            ingredients = ingredients.strip()
        else:
            # Default to Indian if no region specified
            region = "Indian"
            ingredients = query_string.strip()
        
        return self.generate_recipe(region, ingredients)


def main():
    """Example usage"""
    try:
        # Initialize the RAG chain
        rag = RecipeRAGChain()
        
        # Example query
        query = "South American: Grapeseed oil, Simply Potatoes Shredded Hash Browns, Carrots, Green onions, Aged goat cheese, Sun-dried tomato, Eggs, Salt, Black pepper, Butter."
        
        # Generate recipe
        result = rag.query(query)
        
        # Display results
        print("="*50)
        print("GENERATED RECIPE")
        print("="*50)
        print(result["answer"])
        
        print("\n" + "="*50)
        print("SOURCE DOCUMENTS")
        print("="*50)
        for i, doc in enumerate(result["context"], 1):
            print(f"\n--- Source {i} ---")
            print(f"Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"Region: {doc.metadata.get('region', 'Unknown')}")
            print(f"Content Preview: {doc.page_content[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Built the vector database using vector_db_builder.py")
        print("2. Started the Ollama server and pulled the model")


if __name__ == "__main__":
    main()