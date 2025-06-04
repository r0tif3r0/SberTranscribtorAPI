import os
from langchain_gigachat.chat_models import GigaChat
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def get_giga_chat(model_name="GigaChat-2-Max", temp_value=0.87, top_p_value=0.47) -> GigaChat:
    chat_model = GigaChat(
        scope="GIGACHAT_API_CORP",
        credentials=os.environ.get("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        model=model_name,
        temperature=temp_value,
        top_p=top_p_value,
        max_tokens=10000,
        timeout=300,
    )
    return chat_model
