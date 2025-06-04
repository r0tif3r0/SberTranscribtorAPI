import os
from GigaAM import gigaam
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "v2_rnnt"
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

model = gigaam.load_model(MODEL_NAME)

def get_model():
    return model