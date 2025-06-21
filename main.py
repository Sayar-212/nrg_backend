"""
Main Application
Combines all components to create a complete recipe generation system
"""

import sys
import argparse
from pathlib import Path
from ollama_manager import OllamaManager
from vector_db_builder import RecipeVectorDBBuilder
from rag_chain import RecipeRAGChain


class RecipeApp:
    def __init__(self, csv_path=None, model_name="llama3.2:1b"):
        self.csv_path = csv_path
        self.model_name = model_name
        self.ollama_manager = None
        self.db_builder = None
        self.rag_chain = None
        
    def setup_ollama(self):
        """Setup and start Ollama server"""
        print("Setting up Ollama server...")
        self.ollama_manager = OllamaManager()
        
        if not self.ollama_manager.start_server():
            print("Failed to start Ollama server")
            return False
            
        # Pull the model
        if not self.ollama_manager.pull_model(self.model_name):
            print(f"Failed to pull model: {self.model_name}")
            return False
            
        return True
    
    def setup_vector_db(self, force_rebuild=False):
        """Setup vector database"""
        self.db_builder = RecipeVectorDBBuilder()
        
        # Check if database already exists
        db_exists = Path(self.db_builder.persist_directory).exists()
        
        if db_exists and not force_rebuild:
            print("Vector database already exists. Loading existing database...")
            try:
                db = self.db_builder.load_existing_db()
                return True
            except Exception as e:
                print(f"Error loading existing database: {e}")
                print("Will rebuild database...")
        
        # Build new database
        if not self.csv_path:
            print("CSV path required to build vector database")
            return False
            
        if not Path(self.csv_path).exists():
            print(f"CSV file not found: {self.csv_path}")
            return False
            
        try:
            print("Building vector database from CSV...")
            db = self.db_builder.build_from_csv(self.csv_path)
            return True
        except Exception as e:
            print(f"Error building vector database: {e}")
            return False
    
    def setup_rag_chain(self):
        """Setup RAG chain"""
        try:
            self.rag_chain = RecipeRAGChain(model_name=self.model_name)
            return True
        except Exception as e:
            print(f"Error setting up RAG chain: {e}")
            return False
    
    def initialize(self, force_rebuild_db=False):
        """Initialize all components"""
        print("Initializing Recipe Generation System...")
        
        # Setup Ollama
        if not self.setup_ollama():
            return False
            
        # Setup Vector DB
        if not self.setup_vector_db(force_rebuild_db):
            return False
            
        # Setup RAG Chain
        if not self.setup_rag_chain():
            return False
            
        print("System initialized successfully!")
        return True
    
    def generate_recipe(self, query):
        """Generate a recipe from user query"""
        if not self.rag_chain:
            print("System not initialized. Call initialize() first.")
            return None
            
        return self.rag_chain.query(query)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n" + "="*60)
        print("RECIPE GENERATION SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Enter queries in format: 'Region: ingredients'")
        print("Or just 'ingredients' (defaults to Indian cuisine)")
        print("Type 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                query = input("\nEnter your recipe request: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    print("Please enter a valid query")
                    continue
                
                print("\nGenerating recipe...")
                result = self.generate_recipe(query)
                
                if result:
                    print("\n" + "="*50)
                    print("GENERATED RECIPE")
                    print("="*50)
                    print(result["answer"])
                    
                    # Optionally show sources
                    show_sources = input("\nShow source documents? (y/n): ").strip().lower()
                    if show_sources == 'y':
                        print("\n" + "="*50)
                        print("SOURCE DOCUMENTS")
                        print("="*50)
                        for i, doc in enumerate(result["context"], 1):
                            print(f"\n--- Source {i} ---")
                            print(f"Title: {doc.metadata.get('title', 'Unknown')}")
                            print(f"Region: {doc.metadata.get('region', 'Unknown')}")
                            print(f"Content Preview: {doc.page_content[:200]}...")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye!")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.ollama_manager:
            self.ollama_manager.stop_server()


def main():
    parser = argparse.ArgumentParser(description='Recipe Generation System')
    parser.add_argument('--csv', help='Path to recipe CSV file')
    parser.add_argument('--model', default='llama3.2:1b', help='Ollama model name')
    parser.add_argument('--rebuild-db', action='store_true', help='Force rebuild vector database')
    parser.add_argument('--query', help='Single query to process')
    
    args = parser.parse_args()
    
    # Create app instance
    app = RecipeApp(csv_path=args.csv, model_name=args.model)
    
    try:
        # Initialize system
        if not app.initialize(force_rebuild_db=args.rebuild_db):
            print("Failed to initialize system")
            sys.exit(1)
        
        # Process single query or run interactive mode
        if args.query:
            result = app.generate_recipe(args.query)
            if result:
                print("="*50)
                print("GENERATED RECIPE")
                print("="*50)
                print(result["answer"])
        else:
            app.interactive_mode()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
