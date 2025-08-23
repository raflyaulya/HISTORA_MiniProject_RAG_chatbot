import os 
from dotenv import * 
from pathlib import Path
import warnings 
warnings.filterwarnings('ignore')

# Deepseek Part -----------------
load_dotenv(find_dotenv()) 
DEEPSEEK_API = os.getenv('DEEPSEEK_API') 

# Path or Dir -------------------------
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / 'data' 
DB_DIR = BASE_DIR / 'chromadb_store' 

# Embedding path ------------------------
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' 

# Chunking Process -----------------
CHUNK_SIZE= 500
CHUNK_OVERLAP = int(CHUNK_SIZE * (2/10))

