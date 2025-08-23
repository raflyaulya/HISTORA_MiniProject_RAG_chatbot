from langchain_deepseek import *
# from langchain_deepseek.chat_models import ChatDeepSeek
from dotenv import * 
import random
import string 
import os 
import logging 
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

load_dotenv(find_dotenv())

# setup API KEY
DEEPSEEK_API = os.getenv('DEEPSEEK_API')
assert DEEPSEEK_API, 'Set DEEPSEEK_API di .env dulu ya' 

# setup MODEL Deepseek
model_name = 'deepseek-chat'
# model_name = 'deepseek-reasoner'

def llm_deepseek():
    llm = ChatDeepSeek(
        api_key=DEEPSEEK_API, 
        temperature= .5, 
        # max_tokens= max_tokens,
        model=model_name,) 
    
    return llm

# llm = ChatDeepSeek(
#     api_key=DEEPSEEK_API, 
#     temperature= .3, 
#     model=model_name
# )

# ================  GENERATE & GET 1 RANDOM CODE ==> tujuannya untuk persist dir di chroma

n_random_code = 7
total_random_code = n_random_code* 10

def generate_random_code():
    # Gabungan huruf kecil dan angka
    characters = string.ascii_lowercase + string.digits
    # Generate 5 karakter random
    return ''.join(random.choice(characters) for _ in range(n_random_code))

# Generate 5 kode unik
unique_codes = set()
while len(unique_codes) < total_random_code:
    code = generate_random_code()
    unique_codes.add(code)

# Tampilkan hasil
list_of_five_randoms = []
for code in unique_codes:
    # print(code)
    list_of_five_randoms.append(code)

# print(list_of_five_randoms)

# to get one of the code in list 
def pick_1_random_choice():
    code_terpilih = random.choice(list_of_five_randoms)
    return code_terpilih
