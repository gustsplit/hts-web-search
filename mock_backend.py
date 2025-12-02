from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HTSRequest(BaseModel):
    product_name: str
    exporter_hts: Optional[str] = None
    origin_country: Optional[str] = None
    description: str

class HTSItem(BaseModel):
    hts_code: str
    confidence: Optional[float] = None
    title: Optional[str] = None
    tariff: Optional[float] = None
    reason: Optional[str] = None
    source: Optional[str] = None

class HTSResponse(BaseModel):
    items: List[HTSItem]
    query_prompt: str

@app.get("/api/exchange-rate/usd-krw")
def get_usd_krw():
    return {
        "rate": 1400.0,
        "source": "mock",
        "timestamp": "2025-12-02T12:00:00Z"
    }

@app.post("/api/search_hts")
def search_hts(req: HTSRequest):
    return HTSResponse(
        items=[
            HTSItem(hts_code="1005.90.0000", confidence=0.95, title="Corn", tariff=0.05, reason="It is corn"),
            HTSItem(hts_code="2001.90.0000", confidence=0.8, title="Pickled Corn", tariff=0.1, reason="It is pickled")
        ],
        query_prompt="Mock prompt"
    )

@app.post("/api/translate")
def translate(req: dict):
    return {"translated_text": "Title: 옥수수\nReason: 옥수수입니다."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
