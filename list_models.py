import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# 환경변수에서 API 키 불러오기
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 모델 리스트 불러오기
models = genai.list_models()

print("=== Available Models for your API Key ===\n")
for m in models:
    print(m.name)
