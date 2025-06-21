# app.py - Local FastAPI server for RatatouilleGen
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import threading
import subprocess
import requests
import json
import time

# LangChain imports
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

app = FastAPI(title="RatatouilleGen API", version="1.0.0")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class RecipeRequest(BaseModel):
    region: str
    ingredients: str

class RecipeResponse(BaseModel):
    recipe: str
    sources: Optional[list] = None
    status: str

# Global variables
chain = None
ollama_started = False

def start_ollama():
    """Start Ollama server in background"""
    global ollama_started
    if not ollama_started:
        try:
            os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
            os.environ['OLLAMA_ORIGINS'] = '*'
            subprocess.Popen(["ollama", "serve"])
            time.sleep(5)  # Wait for server to start
            ollama_started = True
            print("Ollama server started")
        except Exception as e:
            print(f"Error starting Ollama: {e}")

def load_rag_chain():
    """Initialize the RAG chain"""
    global chain
    
    if chain is None:
        try:
            # Initialize model
            model = ChatOllama(model="qwen3:8b", temperature=0.2)
            
            # Your existing prompt template
            prompt = PromptTemplate.from_template(
                """You are an experienced culinary innovator. Master of making new/novel recipes which are edible and not fully imaginitive.
                Create new , novel, practical and delicious recipes using ONLY the provided ingredients, authentically adapted to the specified regional cuisine style. Nothing else may be added beyond the given ingredients.
                Use the following context from the recipe database to inform your response:
                {context}
                Based on the user's query, identify the region and ingredients, then create a new , novel authentic regional recipe.
                User Query: {question}
                FORMAT (follow exactly):
                Recipe Title:
                The title should be a summary of the dish in max 3 words, also reflecting the region/culture
                Example : if query is India and ingredients are Chicken , Rice , Garam masala
                Recipe title : Bhuna Murg Chawal

                Taste : The taste of the dish should adhere to the preparation. If no sweetenery ingredients is added it should not predict sweet
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
                - The recipe should be new , unorthodox and never seen before type
                - Use ONLY the listed ingredients - no additions, substitutions, or extras
                - Stay authentic to the regional cuisine's core principles and techniques and make new combinations.
                - Create something new yet culturally grounded
                - Focus on emotional satisfaction and cultural resonance
                - The preparation should be stunning. Absolute like a human.
                Recipe:"""
            )
            
            # Initialize embeddings
            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load vector store (you'll need to copy this from Colab)
            vector_store = Chroma(
                persist_directory="./recipe_chroma_db",
                embedding_function=embedding
            )
            
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 3,
                    "score_threshold": 0.6,
                },
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            def enhanced_chain(inputs):
                region = inputs["region"]
                ingredients = inputs["ingredients"]
                formatted_query = f"New recipe from Region: {region},and using these Ingredients: {ingredients}"
                result = qa_chain({"query": formatted_query})
                return {
                    "answer": result["result"],
                    "context": result["source_documents"]
                }
            
            chain = enhanced_chain
            print("RAG chain loaded successfully")
            
        except Exception as e:
            print(f"Error loading RAG chain: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    print("Starting RatatouilleGen API...")
    start_ollama()
    load_rag_chain()

@app.get("/")
async def root():
    return {"message": "RatatouilleGen API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ollama_running": ollama_started,
        "chain_loaded": chain is not None
    }

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """Generate a recipe based on region and ingredients"""
    
    if chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")
    
    try:
        # Generate recipe using your existing logic
        result = chain({
            "region": request.region,
            "ingredients": request.ingredients
        })
        
        # Format sources for frontend
        sources = []
        for doc in result.get("context", []):
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "region": doc.metadata.get("region", "Unknown"),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        return RecipeResponse(
            recipe=result.get("answer", ""),
            sources=sources,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recipe: {str(e)}")

@app.post("/ask")
async def ask_endpoint(request: RecipeRequest):
    """Alternative endpoint matching your original ask function"""
    return await generate_recipe(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)