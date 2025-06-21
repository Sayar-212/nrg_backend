# RatatouilleGen - Recipe Generation System

A Python-based AI system that generates novel recipes using RAG (Retrieval-Augmented Generation) with Ollama and Qwen2 model.

## Features

- **AI-Powered Recipe Creation**: Uses Qwen2:7b model via Ollama for intelligent recipe generation
- **Regional Cuisine Adaptation**: Creates authentic recipes adapted to specific regional cooking styles
- **Vector Database Search**: Uses ChromaDB with HuggingFace embeddings for recipe knowledge retrieval
- **Dual Interface**: Command-line interface and FastAPI web service
- **Automatic Setup**: Handles Ollama installation and model downloading

## System Requirements

- Python 3.8 or higher
- At least 5GB free disk space
- 8GB+ RAM recommended
- Internet connection for initial setup

## Installation

### 1. Clone/Download the Files

Place all the Python files in a directory:
- `main.py` - Command-line interface
- `app.py` - FastAPI web service
- `ollama_manager.py` - Ollama server management
- `vector_db_builder.py` - Vector database builder
- `rag_chain.py` - RAG implementation
- `setup.py` - Setup script
- `requirements.txt` - Python dependencies

### 2. Run Setup

```bash
python setup.py
```

This will:
- Install all Python dependencies
- Create necessary directories
- Set up HuggingFace token (optional)

### 3. Prepare Your Recipe Data

Place your recipe CSV file in the `./data/` directory. The CSV should have these columns:
- `recipe_title`
- `description`
- `region`
- `subregion`
- `ingredients`
- `servings`
- `prep_time`
- `cook_time`
- `steps`
- `author` (optional)
- `vegan` (optional)

## Usage

### Local Development (Command Line)

For local development and testing:

1. Build vector database:
```bash
python vector_db_builder.py --csv ./data/your_recipe_file.csv
```

2. Run in interactive mode:
```bash
python main.py --csv ./data/your_recipe_file.csv
```

3. Run single query:
```bash
python main.py --csv ./data/your_recipe_file.csv --query "Italian: tomatoes, basil, mozzarella, pasta"
```

### FastAPI Web Service Export

For deploying as a web service:

1. Build vector database:
```bash
python vector_db_builder.py --csv ./data/your_recipe_file.csv
```

2. Run FastAPI server:
```bash
python app.py
```

The server will start on `http://localhost:8000` with:
- API documentation at `/docs`
- ReDoc documentation at `/redoc`
- Health check at `/health-check`

### API Endpoints

- `POST /generate-recipe` - Generate a new recipe
  - Request body:
    ```json
    {
      "region": "string",
      "ingredients": "string"
    }
    ```
  - Response:
    ```json
    {
      "recipe": "string",
      "sources": [array of source documents],
      "status": "string"
    }
    ```

- `GET /health-check` - Check server status

### Query Format

Enter queries in one of these formats:
- `Region: ingredients` - e.g., "Italian: tomatoes, basil, pasta"
- `ingredients only` - defaults to Indian cuisine

Examples:
- `South American: Grapeseed oil, Hash Browns, Carrots, Green onions, Goat cheese, Sun-dried tomato, Eggs`
- `Japanese: Rice, Nori, Salmon, Cucumber, Avocado`
- `Chicken, Rice, Garam masala` (defaults to Indian)

## File Structure

```
recipe-system/
├── app.py                 # FastAPI web service
├── main.py               # Command-line interface
├── ollama_manager.py     # Ollama server management
├── vector_db_builder.py  # Builds vector database from CSV
├── rag_chain.py         # RAG implementation
├── setup.py             # Setup script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # Place CSV files here
├── recipe_chroma_db/   # Vector database (auto-created)
└── models/            # Model cache (auto-created)
```

## Command Line Options

### For main.py:
- `--csv PATH`: Path to recipe CSV file
- `--model NAME`: Ollama model name (default: qwen2:7b)
- `--rebuild-db`: Force rebuild of vector database
- `--query TEXT`: Process single query instead of interactive mode

### For app.py:
- No command line arguments needed, runs as a web service
- Configuration is handled through environment variables and code settings

## Troubleshooting

### Ollama Installation Issues

If Ollama fails to install automatically:

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from: https://ollama.com/download

### Model Download Issues

Manually pull the model:
```bash
ollama pull qwen2:7b
```

### Vector Database Issues

Force rebuild the database:
```bash
python main.py --csv ./data/your_file.csv --rebuild-db
```

### Memory Issues

- Use a smaller model: `--model qwen2:1.5b`
- Reduce chunk size in `vector_db_builder.py`
- Close other applications

## Dependencies

See `requirements.txt` for all dependencies. Key components:
- `langchain` - RAG framework
- `chromadb` - Vector database
- `ollama` - AI model serving
- `sentence-transformers` - Text embeddings
- `pandas` - Data processing

## How It Works

1. **Vector Database**: Recipes from CSV are embedded and stored in ChromaDB
2. **Query Processing**: User queries are parsed for region and ingredients
3. **Retrieval**: Similar recipes are found using vector similarity search
4. **Generation**: Qwen2 model creates novel recipes using retrieved context
5. **Output**: Formatted recipes with regional authenticity and cooking instructions

## Output Format

Generated recipes include:
- **Recipe Title**: 3-word summary reflecting regional style
- **Taste Profile**: Expected flavor characteristics
- **Regional Context**: Cultural authenticity explanation
- **Ingredient List**: Precise measurements for 4 servings
- **Preparation Steps**: Detailed cooking instructions
- **Chef's Notes**: Cultural insights and technique explanations

## Contributing

Feel free to modify and extend the system:
- Add new embedding models in `vector_db_builder.py`
- Customize prompts in `rag_chain.py`
- Add new output formats in `main.py`
- Enhance Ollama management in `ollama_manager.py`

## License

Open source - modify and use as needed.