import os
from dotenv import load_dotenv

# Load .env file
load_dotenv('.env')

# Check what we get
search_api_value = os.environ.get('SEARCH_API')
print(f"SEARCH_API from environment: {search_api_value}")
print(f"Type: {type(search_api_value)}")

# Test the configuration loading
from src.open_deep_research.configuration import Configuration, SearchAPI

config = Configuration.from_runnable_config()
print(f"\nConfiguration search_api: {config.search_api}")
print(f"Type: {type(config.search_api)}")
print(f"Value: {config.search_api.value if hasattr(config.search_api, 'value') else config.search_api}")
