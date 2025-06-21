"""
Ollama Server Manager
Handles Ollama installation, server startup, and model management
"""

import os
import subprocess
import threading
import time
import requests
import platform
import sys
from pathlib import Path


class OllamaManager:
    def __init__(self, host='0.0.0.0', port=11434, timeout=300, use_gpu=True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.server_process = None
        self.server_thread = None
        
    def install_ollama(self):
        """Install Ollama based on the operating system"""
        system = platform.system().lower()
        
        try:
            if system == "linux":
                print("Installing Ollama on Linux...")
                subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh"], 
                             stdout=subprocess.PIPE, shell=True, check=True)
                subprocess.run(["sh"], input=requests.get("https://ollama.com/install.sh").text, 
                             text=True, check=True)
            
            elif system == "darwin":  # macOS
                print("Please install Ollama manually from https://ollama.com/download")
                print("Or use: brew install ollama")
                
            elif system == "windows":
                print("Please install Ollama manually from https://ollama.com/download")
                
            else:
                print(f"Unsupported operating system: {system}")
                return False
                
            print("Ollama installation completed!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing Ollama: {e}")
            return False
    
    def check_ollama_installed(self):
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start_server(self):
        """Start Ollama server with GPU if available in a separate thread"""
        def run_server():
            try:
                # Set environment variables
                os.environ['OLLAMA_HOST'] = f'{self.host}:{self.port}'
                os.environ['OLLAMA_ORIGINS'] = '*'
                
                print(f"Starting Ollama server on {self.host}:{self.port}...")
                if self.use_gpu:
                    self.server_process = subprocess.Popen(
                        ["ollama", "serve", "--gpu"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                else:
                    self.server_process = subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                print("Ollama server started successfully!")
                
            except Exception as e:
                print(f"Error starting Ollama server: {e}")
        
        if not self.check_ollama_installed():
            print("Ollama not found. Installing...")
            if not self.install_ollama():
                print("Failed to install Ollama. Please install manually.")
                return False
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait for server to start
        self.wait_for_server()
        return True
    
    def wait_for_server(self, timeout=30):
        """Wait for Ollama server to be ready"""
        print("Waiting for Ollama server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://{self.host}:{self.port}')
                if response.status_code == 200:
                    print("Ollama server is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            time.sleep(1)
        
        print("Timeout waiting for Ollama server to start")
        return False
    
    def pull_model(self, model_name="llama3.2:1b"):
        """Pull a model from Ollama"""
        try:
            print(f"Pulling model: {model_name}...")
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True, text=True, check=True
            )
            print(f"Model {model_name} pulled successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model {model_name}: {e}")
            return False
    
    def stop_server(self):
        """Stop the Ollama server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("Ollama server stopped.")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_server()


if __name__ == "__main__":
    # Tezst the Ollama manager
    with OllamaManager() as ollama:
        # Pull the model
        ollama.pull_model("llama3.2:1b")
        
        # Keep server running for testing
        print("Server is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")