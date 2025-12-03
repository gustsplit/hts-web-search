from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, conint, parse_obj_as
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import os
import json
import re
import time
import logging

import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

from dotenv import load_dotenv
from supabase import create_client, Client

# 이메일 발송용 추가 import
import smtplib
from email.message import EmailMessage

# ================== 로깅 설정 ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ================== FastAPI 기본 설정 ==================

app = FastAPI()


def _parse_origins(raw: Optional[str]) -> List[str]:
    if not raw:
        # 안전한 기본값 (로컬 개발 전용)
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = _parse_origins(os.getenv("ALLOWED_ORIGINS"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, including 'null' (file://)
    allow_credentials=False, # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== .env 로드 추가 ==================
from pathlib import Path
from dotenv import load_dotenv

# 현재 파일(main.py) 기준 backend/.env 파일 경로
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# .env 파일 불러오기
load_dotenv(ENV_PATH)

# 환경변수 읽기
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMAIL_ACCESS_TOKEN = os.getenv("EMAIL_ACCESS_TOKEN")
EMAIL_ALLOWED_DOMAINS = [d.strip() for d in os.getenv("EMAIL_ALLOWED_DOMAINS", "").split(",") if d.strip()]
ENABLE_SEARCH_LOGGING = os.getenv("ENABLE_SEARCH_LOGGING", "false").lower() in {"1", "true", "yes"}

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("REPLACE_") or GOOGLE_API_KEY in ["REDACTED", "PLACEHOLDER"]:
    raise RuntimeError("유효한 GOOGLE_API_KEY가 설정되어 있지 않습니다. 환경변수를 확인해 주세요.")

logger.info("GOOGLE_API_KEY loaded successfully.")

# Google Generative AI SDK 설정
genai.configure(api_key=GOOGLE_API_KEY)

# ================== 데이터 모델 ==================

class HTSRequest(BaseModel):
    product_name: str
    exporter_hs: Optional[str] = None
    description: str
    country_of_origin: str


class HTSExplainRequest(BaseModel):
    hts_code: str


class TranslateRequest(BaseModel):
    text: str


class EmailPreviewRequest(BaseModel):
    """
    HTS / 관세 계산 결과를 기반으로
    사용자에게 발송할 이메일 미리보기를 만드는 요청 모델
    """
    product_name: str
    exporter_hs: Optional[str] = None
    description: str
    country_of_origin: str

    email_to: str

    hts_code: str
    hts_title: Optional[str] = None
    estimated_tariff_rate: Optional[str] = None
    confidence: Optional[int] = None

    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total_value_usd: Optional[float] = None
    duty_usd: Optional[float] = None

    total_value_krw: Optional[float] = None
    duty_krw: Optional[float] = None
    usd_krw_rate: Optional[float] = None


class EmailSendRequest(BaseModel):
    """
    실제 이메일 발송 요청용 모델
    - 프론트에서 미리보기로 받은 subject/body 를 그대로 사용
    """
    email_to: str
    subject: str
    body: str


class HTSCandidate(BaseModel):
    hts_code: str
    title: str
    estimated_tariff_rate: Optional[str]
    reason: Optional[str]
    confidence: Optional[conint(ge=0, le=100)] = None


class HTSSegment(BaseModel):
    code_part: str
    meaning_ko: str


# ================== 공용 유틸 함수 ==================

def extract_json(text: str) -> str:
    """
    LLM이 ```json ... ``` 같은 코드블록으로 응답해도
    실제 JSON 배열 부분만 깔끔하게 잘라서 반환.
    """
    if text is None:
        raise ValueError("LLM response is empty")

    cleaned = text.strip()

    # ```json ... ``` 코드블록 제거
    fence_match = re.search(
        r"```(?:json)?(.*)```",
        cleaned,
        re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # 텍스트 안에 JSON 배열이 섞여 있을 때, 배열 부분만 추출
    array_match = re.search(r"(\[\s*{.*}\s*\])", cleaned, re.DOTALL)
    if array_match:
        cleaned = array_match.group(1).strip()

    return cleaned


def log_search(req: HTSRequest, candidates: List[dict]):
    """
    사용자가 검색한 내용 + LLM이 제안한 HTS 코드 후보를
    Supabase 'search_logs' 테이블에 insert.
    """
    if not ENABLE_SEARCH_LOGGING:
        return

    if not supabase:
        logger.warning("Supabase client not initialized, skipping log.")
        return

    log_entry = {
        "product_name": req.product_name,
        "exporter_hs": req.exporter_hs,
        "description": req.description,
        "country_of_origin": req.country_of_origin,
        "candidates": candidates,  # Supabase JSONB column
        # timestamp is usually handled by default value in DB or we can send it
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        supabase.table("search_logs").insert(log_entry).execute()
    except Exception as e:
        # 로그 실패해도 서비스는 계속
        logger.warning(f"Failed to write log to Supabase: {e}")


# ================== 이메일 발송 유틸 (Gmail) ==================

def send_email_via_gmail(to_email: str, subject: str, body: str):
    """
    Gmail을 이용해 실제 이메일을 발송한다.
    - 발신자: tarfaservice@gmail.com (고정)
    - 비밀번호: 환경변수 GMAIL_APP_PASSWORD (앱 비밀번호 추천)
    """
    gmail_user = "tarfaservice@gmail.com"
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")

    if not gmail_pass:
        raise RuntimeError("GMAIL_APP_PASSWORD 환경변수가 설정되어 있지 않습니다.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg["Reply-To"] = gmail_user
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.send_message(msg)


def _enforce_email_policy(to_email: str, provided_token: Optional[str]):
    """최소한의 남용 방지를 위해 토큰/도메인을 확인한다."""

    if EMAIL_ACCESS_TOKEN:
        if not provided_token:
            raise HTTPException(status_code=401, detail="X-Access-Token header is required for email sending")
        if provided_token != EMAIL_ACCESS_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid access token")

    if "@" not in to_email:
        raise HTTPException(status_code=400, detail="Invalid recipient email")

    if EMAIL_ALLOWED_DOMAINS:
        lowered = to_email.lower()
        if not any(lowered.endswith(f"@{d.lower()}") for d in EMAIL_ALLOWED_DOMAINS):
            raise HTTPException(status_code=400, detail="Recipient domain is not allowed")


# ================== 기본 헬스체크 ==================

@app.get("/")
def read_root():
    return {"message": "HTS Web PoC Backend (Gemini) is running"}


# ================== USITC HTS Verification ==================

def fetch_usitc_codes(hs6: str) -> List[str]:
    """
    USITC API를 호출하여 해당 6자리 HS 코드 하위의 유효한 HTS 코드 목록을 가져옴.
    """
    # 6자리만 남기고 점(.) 등 제거
    clean_hs6 = re.sub(r"[^0-9]", "", hs6)[:6]
    if len(clean_hs6) < 6:
        return []

    # API 호출 파라미터 준비
    # from: 6자리 (예: 847170)
    # to: 6자리 + .99.99 (예: 847170.99.99) - 사용자 요청 포맷 준수
    # 실제 API 동작 확인을 위해 점을 포함한 포맷으로 시도
    # 보통 HTS API는 점이 없는 포맷이나 있는 포맷 둘 다 지원할 수 있으나,
    # 사용자 가이드에 따라 .99.99를 붙임.
    
    # 6자리를 4.2 포맷으로 변환 (예: 8471.70)
    formatted_hs6 = f"{clean_hs6[:4]}.{clean_hs6[4:]}"
    
    # API는 from 파라미터도 10자리 포맷을 선호하는 것으로 보임 (테스트 결과)
    # 예: 8471.70 -> 8471.70.00.00
    from_code = f"{formatted_hs6}.00.00"
    to_code = f"{formatted_hs6}.99.99"
    
    url = "https://hts.usitc.gov/reststop/exportList"
    params = {
        "format": "JSON",
        "from": from_code,
        "to": to_code,
        "styles": "false"
    }
    
    try:
        # 타임아웃 5초 설정
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        # 응답 구조에서 hts_no 추출
        # 응답 예시: [{"htsno": "8471.70.10.00", ...}, ...]
        valid_codes = []
        for item in data:
            code = item.get("htsno")
            if code:
                # 비교를 위해 점 제거한 버전도 고려할 수 있으나,
                # 여기서는 API가 리턴한 포맷(보통 점 포함)을 그대로 리스트업
                valid_codes.append(code)
                
        return valid_codes
        
    except Exception as e:
        logger.warning(f"Failed to fetch USITC codes for {hs6}: {e}")
        return []

# ================== HTS 코드 추천 API ==================

@app.post("/api/search_hts")
def search_hts(req: HTSRequest):
    """
    LLM(Gemini)을 사용해서 US HTS 코드 후보를 3~5개 추천.
    """
    prompt = f"""
You are a US customs classification assistant.

Based on the following information, suggest 3 to 5 candidate US HTS codes.

Information:
- Product name: {req.product_name}
- Exporter HS code (Korea): {req.exporter_hs}
- Product description: {req.description}
- Country of origin: {req.country_of_origin}

For each candidate, return a JSON object with:
- hts_code (string)
- title (short HTS description)
- estimated_tariff_rate (e.g. "0%", "4%", or "N/A")
- reason (short explanation in English)
- confidence (integer 0–100, how likely this HTS code is correct)

Return ONLY a JSON array.
DO NOT add any markdown code fences like ``` or ```json.
DO NOT add any explanation before or after the JSON.
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)

        raw_text = response.text or ""
        cleaned_json_str = extract_json(raw_text)

        candidates_raw = json.loads(cleaned_json_str)
        validated = parse_obj_as(List[HTSCandidate], candidates_raw)
        candidates = [c.dict() for c in validated]

        # ---------------------------------------------------------
        # [검증 로직 추가]
        # 1. 후보군 중 USITC에 실제로 존재하지 않는 코드가 있는지 확인
        # 2. 없으면 해당 6자리 대역의 유효 코드를 조회하여 재검색(Re-prompt)
        # ---------------------------------------------------------
        
        # 검증 대상 6자리 코드 수집
        hs6_map = {} # { "847170": ["8471.70.10.00", ...] }
        
        needs_correction = False
        
        for cand in candidates:
            code = cand["hts_code"]
            clean_code = re.sub(r"[^0-9]", "", code)
            
            # 6자리 미만이면 검증 불가 (혹은 잘못된 코드)
            if len(clean_code) < 6:
                continue
                
            hs6 = clean_code[:6]
            
            # 이미 조회한 적 없으면 API 호출
            if hs6 not in hs6_map:
                valid_list = fetch_usitc_codes(hs6)
                # API 호출 실패시 빈 리스트일 수 있음 -> 이 경우 검증 스킵(보수적 접근)
                # 하지만 빈 리스트가 '진짜 없음'일 수도 있음. 
                # 여기서는 API 오류가 아니라면 신뢰한다고 가정.
                hs6_map[hs6] = valid_list
            
            # 유효성 체크
            # API에서 가져온 코드는 점이 있을 수 있음. 비교를 위해 둘 다 점 제거 후 비교
            valid_clean_set = {re.sub(r"[^0-9]", "", vc) for vc in hs6_map[hs6]}
            
            if clean_code not in valid_clean_set and hs6_map[hs6]:
                # 유효한 코드가 있는데, 추천된 코드가 그 안에 없다면 -> 없는 코드
                needs_correction = True
                cand["_invalid"] = True # 마킹
                
        if needs_correction:
            logger.info("Invalid HTS codes detected. Re-prompting with valid codes...")
            
            # 재질문 프롬프트 구성
            # 유효하지 않은 코드가 포함된 6자리 대역의 '실제 존재하는 코드' 목록을 제공
            
            correction_context = ""
            for hs6, valid_codes in hs6_map.items():
                if not valid_codes:
                    continue
                # 해당 6자리 대역의 유효 코드 나열
                correction_context += f"\nValid codes for HS {hs6}: {', '.join(valid_codes)}"
            
            re_prompt = f"""
The previous suggestions included invalid HTS codes that do not exist in the USITC database.
Please re-evaluate and suggest the best 3 to 5 HTS codes ONLY from the valid codes listed below.

{correction_context}

Original Information:
- Product name: {req.product_name}
- Description: {req.description}

For each candidate, return a JSON object with:
- hts_code (string)
- title (short HTS description)
- estimated_tariff_rate (e.g. "0%", "4%", or "N/A")
- reason (short explanation in English)
- confidence (integer 0–100, how likely this HTS code is correct)

Return ONLY a JSON array.
DO NOT add any markdown code fences like ``` or ```json.
DO NOT add any explanation before or after the JSON.
"""
            # 재호출
            response_retry = model.generate_content(re_prompt)
            raw_text_retry = response_retry.text or ""
            cleaned_json_retry = extract_json(raw_text_retry)
            
            candidates_retry_raw = json.loads(cleaned_json_retry)
            validated_retry = parse_obj_as(List[HTSCandidate], candidates_retry_raw)
            candidates = [c.dict() for c in validated_retry]
            
            logger.info("Re-prompt successful. Updated candidates.")

        # ---------------------------------------------------------

        # 검색 로그 저장
        log_search(req, candidates)

        return {"candidates": candidates}

    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON from LLM")
        return {
            "error": "Failed to parse JSON from LLM",
            "raw": raw_text if "raw_text" in locals() else None,
        }
    except ValidationError as e:
        logger.exception("LLM response validation failed in /api/search_hts")
        return {
            "error": "LLM response validation failed",
            "details": e.errors(),
        }
    except Exception as e:
        logger.exception("Error in /api/search_hts")
        return {"error": str(e)}


# ================== translate_to_korean ==================

@app.post("/api/translate")
def translate_text(req: TranslateRequest):
    """
    Title/Reason 영문을 한국어로 자연스럽게 번역.
    HTS, Tariff, Duty, SSD 같은 전문 용어는 그대로 유지하도록 프롬프트 설정.
    """
    prompt = f"""
다음 HTS 관련 설명을 한국어로 자연스럽게 번역해줘. 전체적으로 영어의 비율이 70%가 넘지 않는지를 한번 더 검증해야돼

[번역 규칙]
- 'HTS', 'Tariff', 'Duty' 등
  관세 품목분류 관련 전문 용어는 그대로 영어로 남겨도 좋음.
- 한국 관세/물류 담당자가 읽기 편한 자연스러운 문장으로 바꿔줘.
- 입력 형식은 'Title: ...', 'Reason: ...' 이고,
  출력도 동일하게 'Title: ...', 'Reason: ...' 형식을 유지해줘.

입력:
{req.text}
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        return {"translated_text": (response.text or "").strip()}
    except Exception as e:
        logger.exception("Error in /api/translate")
        return {"error": str(e)}


# ================== USD-KRW 환율 조회 API (12시간 캐시 버전) ==================

_fx_cached_rate: Optional[float] = None
_fx_cached_timestamp: float = 0.0  # epoch time (초 단위)


def _build_fx_response():
    global _fx_cached_rate, _fx_cached_timestamp

    try:
        now = time.time()
        twelve_hours = 12 * 60 * 60

        if _fx_cached_rate is not None and (now - _fx_cached_timestamp) < twelve_hours:
            # Return cached rate; include timestamp (ISO) for readability
            return {
                "rate": _fx_cached_rate,
                "source": "cache",
                "last_update": _fx_cached_timestamp,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        # 무료 환율 API로 교체: https://open.er-api.com
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=5,
        )
        resp.raise_for_status()

        data = resp.json()
        rate = data.get("rates", {}).get("KRW")

        if rate is None:
            raise ValueError(f"KRW rate not found in response: {data}")

        _fx_cached_rate = float(rate)
        _fx_cached_timestamp = now

        return {
            "rate": _fx_cached_rate,
            "source": "live-update",
            "last_update": _fx_cached_timestamp,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print("[FX ERROR]", e)
        if _fx_cached_rate is not None:
            return {
                "rate": _fx_cached_rate,
                "source": "fallback-cache",
                "error": str(e),
                "last_update": _fx_cached_timestamp,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        return {
            "rate": 1400.0,
            "source": "fallback-fixed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


@app.get("/api/exchange-rate/usd-krw")
def get_usd_krw_new():
    return _build_fx_response()


@app.get("/api/usd-krw")
def get_usd_krw_deprecated():
    resp = _build_fx_response()
    resp["deprecated"] = True
    resp["deprecated_message"] = "This endpoint is deprecated; use /api/exchange-rate/usd-krw instead."
    return resp


# ================== HTS 코드 구조 설명 API ==================

@app.post("/api/explain_hts")
def explain_hts(req: HTSExplainRequest):
    """
    HTS 코드(예: 8471704060)를 받아서,
    각 단계별(4자리, 6자리, 8자리, 10자리 등) 코드 구조와 의미를
    한국어로 설명하는 정보를 반환.
    """
    prompt = f"""
너는 미국 HS/HTS 코드 구조를 설명하는 관세 전문가야.

다음 HTS 코드의 계층 구조를 한국어로 설명해줘.

HTS 코드: {req.hts_code}

아래와 같은 형식의 JSON 배열만 반환해.
각 요소는 코드 구조의 한 단계(예: 8471, 8471.70, 8471.70.40, 8471.70.40.60)를 의미해.

반드시 이 형식만 지키고, JSON 이외의 텍스트는 절대 추가하지 마.
(코드 블록 마크다운 ``` 같은 것도 사용하지 말 것.)

[
  {{
    "code_part": "8471",
    "meaning_ko": "자동 데이터 처리(ADP) 장치 및 그 부품(컴퓨터/서버 관련 장비)"
  }},
  {{
    "code_part": "8471.70",
    "meaning_ko": "저장장치(Storage units)"
  }},
  {{
    "code_part": "8471.70.40",
    "meaning_ko": "자기식 디스크 드라이브(HDD)"
  }},
  {{
    "code_part": "8471.70.40.60",
    "meaning_ko": "기타(단품 HDD) – 시스템과 함께 수입되지 않은 HDD"
  }}
]

위 예시는 8471.70.40.60에 대한 예시이고,
실제 응답은 {req.hts_code}에 맞는 code_part / meaning_ko로 구성해줘.
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)

        raw_text = response.text or ""
        cleaned_json_str = extract_json(raw_text)
        segments_raw = json.loads(cleaned_json_str)
        segments = parse_obj_as(List[HTSSegment], segments_raw)

        return {"segments": [s.dict() for s in segments]}

    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON from LLM in /api/explain_hts")
        return {
            "error": "Failed to parse JSON from LLM",
            "raw": raw_text if "raw_text" in locals() else None,
        }
    except ValidationError as e:
        logger.exception("LLM response validation failed in /api/explain_hts")
        return {
            "error": "LLM response validation failed",
            "details": e.errors(),
        }
    except Exception as e:
        logger.exception("Error in /api/explain_hts")
        return {"error": str(e)}


# ================== 이메일 미리보기 API ==================

@app.post("/api/preview_email")
def preview_email(req: EmailPreviewRequest):
    """
    사용자가 검색/선택/계산한 내용을 바탕으로
    Toss Bank 톤에 가까운 안내 메일 초안을 생성해서 반환.
    실제 메일 발송은 하지 않고, subject/body만 내려준다.
    """
    exporter_hs_display = req.exporter_hs.strip() if req.exporter_hs else "(미입력)"

    def fmt_money(v: Optional[float], currency: str = "USD") -> str:
        if v is None:
            return "-"
        try:
            if currency.upper() == "USD":
                return f"{v:,.2f} USD"
            else:
                # KRW
                return f"{int(round(v)):,.0f}원"
        except Exception:
            return str(v)

    subject = f"[US HTS 안내] {req.product_name} 관세 계산 결과를 보내드려요"

    # 본문 구성 (Toss 뱅크 느낌의 짧은 문장 + 정보 정리)
    lines = []

    lines.append(f"{req.email_to} 고객님,")
    lines.append("")
    lines.append("US HTS 코드와 관세 계산 결과를 한 번에 정리해서 보내드렸어요.")
    lines.append("실제 신고 시에는 반드시 최종 관세사 / 세관 판단을 다시 확인해 주세요.")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append("1. 기본 상품 정보")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append(f"- 제품명 : {req.product_name}")
    lines.append(f"- 수출자 HS 코드 : {exporter_hs_display}")
    lines.append(f"- 원산지 : {req.country_of_origin}")
    lines.append(f"- 제품 설명 : {req.description}")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append("2. 선택하신 US HTS 코드")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append(f"- HTS 코드 : {req.hts_code}")
    if req.hts_title:
        lines.append(f"- 품목명(Title) : {req.hts_title}")
    if req.estimated_tariff_rate:
        lines.append(f"- 예상 관세율 : {req.estimated_tariff_rate}")
    if req.confidence is not None:
        lines.append(f"- 모델 신뢰도(참고용) : {req.confidence}%")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━")
    lines.append("3. 관세 시뮬레이션")
    lines.append("━━━━━━━━━━━━━━━")

    if req.qty is not None and req.unit_price is not None:
        lines.append(f"- 수량 : {req.qty:.0f} EA")
        lines.append(f"- 단가 : {fmt_money(req.unit_price, 'USD')}")
    if req.total_value_usd is not None:
        lines.append(f"- 과세가격 합계 : {fmt_money(req.total_value_usd, 'USD')}")
    if req.duty_usd is not None:
        lines.append(f"- 예상 관세 : {fmt_money(req.duty_usd, 'USD')}")

    if req.usd_krw_rate and req.total_value_krw is not None and req.duty_krw is not None:
        lines.append("")
        lines.append(f"(환율 기준 : 1 USD ≈ {req.usd_krw_rate:,.2f}원)")
        lines.append(f"- 과세가격(원화) : {fmt_money(req.total_value_krw, 'KRW')}")
        lines.append(f"- 예상 관세(원화) : {fmt_money(req.duty_krw, 'KRW')}")

    lines.append("")
    lines.append("※ 안내 드린 내용은 PoC 시스템에서 자동 계산된 값으로,")
    lines.append("   실제 신고·납부 금액과는 차이가 있을 수 있습니다.")
    lines.append("   최종 판단은 항상 CBP 및 관세 전문가의 리뷰를 거쳐 주세요.")
    lines.append("")
    lines.append("궁금한 점이 있으면 언제든지 회신으로 편하게 문의 주세요.")
    lines.append("")
    lines.append("감사합니다.")

    body = "\n".join(lines)

    return {
        "to": req.email_to,
        "subject": subject,
        "body": body,
    }


# ================== 이메일 발송 API ==================

@app.post("/api/send_email")
def send_email(req: EmailSendRequest, x_access_token: Optional[str] = Header(None)):
    """
    프론트에서 전달한 subject/body를 그대로 사용해서
    tarfaservice@gmail.com 계정으로 실제 발송.
    """
    _enforce_email_policy(req.email_to, x_access_token)

    try:
        send_email_via_gmail(req.email_to, req.subject, req.body)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/send_email")
        return {"error": str(e)}


# ===== (선택) 로컬에서 직접 테스트용 엔트리 포인트 =====
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",        # 이 파일 안의 app 객체를 사용
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
