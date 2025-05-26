"""
Setup script for the E-commerce Customer Service Chatbot (Ollama Local Version)
"""
import os
import subprocess
import sys
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist (Ollama version)"""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("""# Weather API Configuration (Optional - OpenWeatherMap)
WEATHER_API_KEY=your_weather_api_key_here

# Application Configuration
APP_NAME=E-Commerce Customer Service Bot
DEBUG=True

# Ollama LLM Model (must be pulled locally, e.g. llama3.1)
OLLAMA_MODEL=llama3.1
""")
        print("✅ .env file created. Please add your weather API key if needed.")
    else:
        print("✅ .env file already exists.")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        "logs",
        "data",
        "tests",
        "docs"
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_ollama_running():
    """Check if Ollama is running locally"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Ollama is running locally.")
            return True
        else:
            print("⚠️  Ollama server did not respond as expected.")
            return False
    except Exception as e:
        print(f"❌ Ollama is not running or not reachable at http://localhost:11434. Error: {e}")
        print("➡️  Please install Ollama (https://ollama.com/download), pull your model (e.g. `ollama pull llama3.1`), and run `ollama serve`.")
        return False

def check_api_keys():
    """Check if Weather API key is configured (optional)"""
    from dotenv import load_dotenv
    load_dotenv()
    weather_key = os.getenv("WEATHER_API_KEY")
    if not weather_key or weather_key == "your_weather_api_key_here":
        print("⚠️  Weather API key not configured (optional). Weather features will use mock data.")
    else:
        print("✅ Weather API key is configured.")
    return True

def run_tests():
    """Run basic tests to ensure everything is working"""
    print("Running basic tests...")
    try:
        from mock_databases import MockOrderDatabase, MockProductDatabase, MockCustomerDatabase
        from tools import get_tools
        from agent import OllamaLocalAgent

        print("✅ All imports successful.")

        # Test database initialization
        order_db = MockOrderDatabase()
        product_db = MockProductDatabase()
        customer_db = MockCustomerDatabase()
        print("✅ Mock databases initialized.")

        # Test tools
        tools = get_tools()
        print(f"✅ {len(tools)} tools loaded successfully.")

        # Test Ollama agent initialization
        agent = OllamaLocalAgent(model_name=os.getenv("OLLAMA_MODEL", "llama3.1"))
        print("✅ OllamaLocalAgent initialized.")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up E-commerce Customer Service Chatbot (Ollama Local)...")
    print("=" * 50)

    # Create project structure
    create_project_structure()

    # Create .env file
    create_env_file()

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation.")
        return

    # Check Ollama running
    if not check_ollama_running():
        print("❌ Ollama must be installed, the model pulled, and the server running before use.")
        return

    # Check API keys (weather, optional)
    check_api_keys()

    # Run tests
    if run_tests():
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. (Optional) Update your .env file with a valid WEATHER_API_KEY")
    print("2. Make sure Ollama is running: `ollama serve`")
    print("3. Pull your model: `ollama pull llama3.1`")
    print("4. Run the application: streamlit run app.py")
    print("5. Open your browser to the provided URL")
    print("\nFor help, check the README.md file.")

if __name__ == "__main__":
    main()
