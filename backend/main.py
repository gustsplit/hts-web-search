import json
import logging
import os
import re
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional

import smtplib
import google.generativeai as genai
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, confloat, parse_obj_as
from dotenv import load_dotenv


# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMAIL_ACCESS_TOKEN = os.getenv("EMAIL_ACCESS_TOKEN")
EMAIL_ALLOWED_DOMAINS = [d.strip() for d in os.getenv("EMAIL_ALLOWED_DOMAINS", "").split(",") if d.strip()]
ENABLE_SEARCH_LOGGING = os.getenv("ENABLE_SEARCH_LOGGING", "false").lower() in {"1", "true", "yes"}
GEMINI_MODEL = "gemini-flash-latest"
USITC_API_URL = "https://hts.usitc.gov/api/search"
FX_API_URL = "https://open.er-api.com/v6/latest/USD"
FX_CACHE_SECONDS = 12 * 60 * 60

if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("REPLACE_") or GOOGLE_API_KEY in {"REDACTED", "PLACEHOLDER"}:
    raise RuntimeError("GOOGLE_API_KEY is missing or invalid.")

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
class HTSRequest(BaseModel):
    product_name: str
    exporter_hts: Optional[str] = None
    origin_country: Optional[str] = None
    description: str


class HTSCandidate(BaseModel):
    hts_code: str
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    title: Optional[str] = None
    tariff: Optional[float] = None
    reason: Optional[str] = None


class HTSItem(BaseModel):
    hts_code: str
    confidence: Optional[float] = None
    title: Optional[str] = None
    tariff: Optional[float] = None
    reason: Optional[str] = None
    source: Optional[str] = None  # "gemini" or "usitc_prefix"


class HTSResponse(BaseModel):
    items: List[HTSItem]
    raw_gemini_items: List[HTSItem] = []
    invalid_items: List[HTSItem] = []
    query_prompt: str


class HTSExplainRequest(BaseModel):
    hts_code: str


class TranslateRequest(BaseModel):
    text: str


class EmailPreviewRequest(BaseModel):
    product_name: str
    exporter_hts: Optional[str] = None
    description: str
    country_of_origin: str
    email_to: str
    hts_code: str
    hts_title: Optional[str] = None
    estimated_tariff_rate: Optional[str] = None
    confidence: Optional[float] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total_value_usd: Optional[float] = None
    duty_usd: Optional[float] = None
    total_value_krw: Optional[float] = None
    duty_krw: Optional[float] = None
    usd_krw_rate: Optional[float] = None
    query_prompt: Optional[str] = None


class EmailSendRequest(BaseModel):
    email_to: str
    subject: str
    body: str


class HTSSegment(BaseModel):
    code_part: str
    meaning_ko: str


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def extract_json(text: str) -> str:
    """Strip fences and extract JSON array text."""
    if text is None:
        raise ValueError("LLM response is empty")
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?(.*)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    array_match = re.search(r"(\[\s*{.*}\s*\])", cleaned, re.DOTALL)
    if array_match:
        cleaned = array_match.group(1).strip()
    return cleaned


def extract_json_array(raw_text: str) -> str:
    """
    Try to extract a JSON array substring from the raw Gemini text.
    - If there are ```json ... ``` fences, take the content between them.
    - Otherwise, take the substring from the first '[' to the last ']'.
    - If nothing reasonable is found, raise ValueError.
    """
    if not raw_text:
        raise ValueError("Empty raw_text")

    # Handle fenced blocks
    if "```" in raw_text:
        first = raw_text.find("```")
        last = raw_text.rfind("```")
        if first != -1 and last != -1 and last > first:
            inner = raw_text[first + 3 : last]
            inner = inner.lstrip().lstrip("json").lstrip()
            raw_text = inner

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON array brackets in raw_text: {raw_text!r}")

    return raw_text[start : end + 1]


def parse_markdown_hts_table(raw_text: str) -> List[dict]:
    """
    Extract rows from a Markdown table with header:
    | hts_code | confidence | title | tariff | reason |
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("| hts_code | confidence | title | tariff | reason |"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("HTS markdown table header not found")

    rows: List[dict] = []
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        hts_code = parts[0].replace("**", "").strip()
        conf_str = parts[1].replace("**", "").strip()
        title = parts[2].replace("**", "").strip()
        tariff_raw = parts[3].replace("**", "").strip()
        reason = parts[4].replace("**", "").strip()

        try:
            confidence = float(conf_str)
            if confidence > 1:
                confidence = confidence / 100.0
        except Exception:
            confidence = None

        tariff: Optional[float]
        if tariff_raw.lower() == "free":
            tariff = 0.0
        else:
            try:
                tariff = float(tariff_raw.replace("%", "").strip())
            except Exception:
                tariff = None

        rows.append(
            {
                "hts_code": hts_code,
                "confidence": confidence,
                "title": title,
                "tariff": tariff,
                "reason": reason,
            }
        )
    return rows


def normalize_hts_code(hts_code: Optional[str]) -> str:
    return re.sub(r"\D", "", hts_code or "")


def format_hts_prefix(hts_code: str) -> Optional[str]:
    digits = normalize_hts_code(hts_code)
    if len(digits) < 6:
        return None
    return f"{digits[:4]}.{digits[4:6]}"


def build_usitc_url(hts_code: Optional[str]) -> str:
    cleaned = normalize_hts_code(hts_code)
    if not cleaned:
        return "https://hts.usitc.gov/"
    return f"https://hts.usitc.gov/search?query={cleaned}"


def build_prompt(req: HTSRequest) -> str:
    exporter = req.exporter_hts or "N/A"
    origin = req.origin_country or "N/A"
    prompt_text = f"""
You are an expert on US HTS classification.

Product name: {req.product_name}
Exporter HS/HTS code: {exporter}
Country of origin: {origin}
Product description:
{req.description}

Return 3~5 candidate US HTS codes with:
- hts_code (10 digits)
- confidence (0~1)
- title (short description)
- tariff (duty rate, if known)
- reason (why you chose this code; in Korean, short).

Please respond with a Markdown table only, with columns exactly:
| hts_code | confidence | title | tariff | reason |
Do NOT add any extra commentary or markdown outside the table. If you have no candidates, return an empty JSON array: [].
"""
    return prompt_text.strip()


def fetch_hts_from_usitc(code_or_prefix: str) -> List[dict]:
    """Query USITC HTS API for a code or prefix and flatten results."""
    resp = requests.get(
        USITC_API_URL,
        params={"query": code_or_prefix},
        headers={"Accept": "application/json"},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()
    hits: List[dict] = []

    def _collect(obj):
        if isinstance(obj, dict):
            if any(k in obj for k in ("htsno", "htsno_formatted", "number", "display")):
                hits.append(obj)
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(data)
    return hits


def _hit_code(hit: dict) -> Optional[str]:
    return hit.get("htsno_formatted") or hit.get("htsno") or hit.get("number") or hit.get("display")


def _hit_title(hit: dict) -> Optional[str]:
    return hit.get("heading") or hit.get("description") or hit.get("annotation") or hit.get("heading_short")


def _hit_tariff(hit: dict) -> Optional[float]:
    for key in ("general", "tariff", "rate"):
        if key in hit:
            try:
                return float(str(hit[key]).replace("%", "").strip())
            except Exception:
                return None
    return None


def _is_bookmarked(hit: dict) -> bool:
    return bool(hit.get("bookmark") or hit.get("bookmarked") or hit.get("is_bookmark"))


def log_search(req: HTSRequest, prompt_text: str, raw_items: List[HTSItem], final_items: List[HTSItem], invalid_items: List[HTSItem], raw_text: Optional[str] = None):
    if not ENABLE_SEARCH_LOGGING:
        return
    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOG_DIR / "search_logs.jsonl"
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "product_name": req.product_name,
        "exporter_hts": req.exporter_hts,
        "origin_country": req.origin_country,
        "description": req.description,
        "prompt_text": prompt_text,
        "raw_gemini_items": [i.dict() for i in raw_items],
        "items": [i.dict() for i in final_items],
        "invalid_items": [i.dict() for i in invalid_items],
        "raw_text": (raw_text[:2000] + "…") if raw_text and len(raw_text) > 2000 else raw_text,
    }
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write log: {e}")


# ------------------------------------------------------------
# Email helpers
# ------------------------------------------------------------
def send_email_via_gmail(to_email: str, subject: str, body: str):
    gmail_user = "tarfaservice@gmail.com"
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_pass:
        raise RuntimeError("GMAIL_APP_PASSWORD env var is required for email sending.")
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


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "HTS Web PoC Backend (Gemini) is running"}


def _call_gemini(prompt_text: str) -> tuple[List[HTSCandidate], str]:
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt_text)
    raw_text = response.text or ""
    logger.info("Gemini raw response: %s", raw_text)
    try:
        parsed_rows = parse_markdown_hts_table(raw_text)
    except Exception as e:
        logger.error("Gemini parsing failed: %s\nPrompt: %s\nRaw text: %r", e, prompt_text, raw_text)
        return [], raw_text

    try:
        validated = parse_obj_as(List[HTSCandidate], parsed_rows)
    except ValidationError as e:
        logger.error("Gemini response validation failed: %s\nPrompt: %s\nParsed rows: %r", e, prompt_text, parsed_rows)
        return [], raw_text
    return validated, raw_text


@app.post("/api/search_hts", response_model=HTSResponse)
def search_hts(req: HTSRequest):
    prompt_text = build_prompt(req)
    raw_text: str = ""
    raw_items: List[HTSItem] = []
    final_items: List[HTSItem] = []
    invalid_items: List[HTSItem] = []
    prefix_cache: dict[str, List[dict]] = {}

    try:
        candidates, raw_text = _call_gemini(prompt_text)
        for c in candidates:
            conf_val = None
            if c.confidence is not None:
                conf_val = float(c.confidence)
                if conf_val > 1:
                    conf_val = conf_val / 100.0
            raw_items.append(
                HTSItem(
                    hts_code=c.hts_code,
                    confidence=conf_val,
                    title=c.title,
                    tariff=c.tariff,
                    reason=c.reason,
                    source="gemini",
                )
            )
    except Exception as e:
        logger.exception("Gemini call failed")
        return {"error": str(e), "query_prompt": prompt_text, "items": [], "raw_gemini_items": [], "invalid_items": []}

    seen_codes = set()
    for item in raw_items:
        norm = normalize_hts_code(item.hts_code)
        if not norm:
            invalid_items.append(item)
            # Keep unverified candidate
            final_items.append(item)
            continue

        hit_match = None
        try:
            hits = fetch_hts_from_usitc(item.hts_code)
        except Exception as e:
            logger.warning(f"USITC lookup failed for {item.hts_code}: {e}")
            hits = []

        for h in hits:
            code_candidate = normalize_hts_code(_hit_code(h))
            if not code_candidate:
                continue
            if code_candidate == norm and _is_bookmarked(h):
                hit_match = h
                break

        if hit_match:
            if norm not in seen_codes:
                seen_codes.add(norm)
                item.title = item.title or _hit_title(hit_match)
                item.tariff = _hit_tariff(hit_match)
                final_items.append(item)
                # mark as verified hit
                item.source = item.source or "gemini"
            continue

        invalid_items.append(item)
        prefix = format_hts_prefix(item.hts_code)
        if not prefix:
            # keep original even if no prefix
            final_items.append(item)
            continue
        prefix_digits = normalize_hts_code(prefix)
        try:
            if prefix in prefix_cache:
                prefix_hits = prefix_cache[prefix]
            else:
                prefix_hits = fetch_hts_from_usitc(prefix)
                prefix_cache[prefix] = prefix_hits
        except Exception as e:
            logger.warning(f"USITC prefix lookup failed for {prefix}: {e}")
            # keep original candidate when lookup fails
            final_items.append(item)
            continue

        added_from_prefix = False
        for h in prefix_hits:
            hit_code_val = _hit_code(h)
            norm_hit = normalize_hts_code(hit_code_val)
            if not norm_hit or not norm_hit.startswith(prefix_digits):
                continue
            if not _is_bookmarked(h):
                continue
            if norm_hit in seen_codes:
                continue
            seen_codes.add(norm_hit)
            final_items.append(
                HTSItem(
                    hts_code=hit_code_val or prefix,
                    confidence=item.confidence,
                    title=_hit_title(h),
                    tariff=_hit_tariff(h),
                    reason=item.reason,
                    source="usitc_prefix",
                )
            )
            added_from_prefix = True

        # If nothing added from prefix, keep original unverified candidate
        if not added_from_prefix and item not in final_items:
            final_items.append(item)

    response_body = HTSResponse(
        items=final_items,
        raw_gemini_items=raw_items,
        invalid_items=invalid_items,
        query_prompt=prompt_text,
    )

    log_search(req, prompt_text, raw_items, final_items, invalid_items, raw_text)
    return response_body


@app.post("/api/translate")
def translate_text(req: TranslateRequest):
    prompt = f"""
다음 HTS 관세 설명을 한국어로 자연스럽게 번역해주세요.
출력은 'Title: ...', 'Reason: ...' 형식을 유지해주세요.

입력:
{req.text}
"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return {"translated_text": (response.text or "").strip()}
    except Exception as e:
        logger.exception("Error in /api/translate")
        return {"error": str(e)}


# FX cache
_fx_cached_rate: Optional[float] = None
_fx_cached_timestamp: float = 0.0


def _build_fx_response():
    global _fx_cached_rate, _fx_cached_timestamp
    try:
        now = time.time()
        if _fx_cached_rate is not None and (now - _fx_cached_timestamp) < FX_CACHE_SECONDS:
            return {
                "rate": _fx_cached_rate,
                "source": "cache",
                "last_update": _fx_cached_timestamp,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        resp = requests.get(FX_API_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("rates", {}).get("KRW")
        if rate is None:
            raise ValueError("KRW rate not found")
        _fx_cached_rate = float(rate)
        _fx_cached_timestamp = now
        return {
            "rate": _fx_cached_rate,
            "source": "live-update",
            "last_update": _fx_cached_timestamp,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
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


@app.post("/api/explain_hts")
def explain_hts(req: HTSExplainRequest):
    prompt = f"""
당신은 미국 HS/HTS 코드 구조를 설명하는 관세 전문가입니다.
아래 HTS 코드의 계층 구조를 한국어로 설명하는 JSON 배열을 반환하세요.
HTS 코드: {req.hts_code}
"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        raw_text = response.text or ""
        cleaned_json_str = extract_json(raw_text)
        segments_raw = json.loads(cleaned_json_str)
        segments = parse_obj_as(List[HTSSegment], segments_raw)
        return {"segments": [s.dict() for s in segments]}
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON from LLM in /api/explain_hts")
        return {"error": "Failed to parse JSON from LLM", "raw": raw_text if "raw_text" in locals() else None}
    except ValidationError as e:
        logger.exception("LLM response validation failed in /api/explain_hts")
        return {"error": "LLM response validation failed", "details": e.errors()}
    except Exception as e:
        logger.exception("Error in /api/explain_hts")
        return {"error": str(e)}


@app.post("/api/preview_email")
def preview_email(req: EmailPreviewRequest):
    exporter_hs_display = req.exporter_hs.strip() if req.exporter_hs else "(미입력)"

    def fmt_money(v: Optional[float], currency: str = "USD") -> str:
        if v is None:
            return "-"
        try:
            if currency.upper() == "USD":
                return f"{v:,.2f} USD"
            return f"{int(round(v)):,.0f}원"
        except Exception:
            return str(v)

    subject = f"[US HTS 안내] {req.product_name} 관세 계산 결과를 보내드립니다"
    lines = []
    lines.append(f"{req.email_to} 고객님")
    lines.append("")
    lines.append("US HTS 코드와 관세 계산 결과를 정리해서 보내드렸어요.")
    lines.append("최종 통관 시에는 반드시 관세사 검토를 거쳐 주세요.")
    lines.append("")
    lines.append("────────────")
    lines.append("1. 기본 상품 정보")
    lines.append("────────────")
    lines.append(f"- 상품명: {req.product_name}")
    lines.append(f"- 수출자 HS 코드 : {exporter_hs_display}")
    lines.append(f"- 원산지 : {req.country_of_origin}")
    lines.append(f"- 상품 설명 : {req.description}")
    lines.append("")
    lines.append("────────────")
    lines.append("2. 선택하신 US HTS 코드")
    lines.append("────────────")
    lines.append(f"- HTS 코드 : {req.hts_code}")
    if req.hts_title:
        lines.append(f"- 품목(Title) : {req.hts_title}")
    if req.estimated_tariff_rate:
        lines.append(f"- 예상 관세율 : {req.estimated_tariff_rate}")
    if req.confidence is not None:
        lines.append(f"- 모델 신뢰도(참고용) : {req.confidence * 100:.0f}%")
    lines.append("")
    lines.append("────────────")
    lines.append("3. 관세 시뮬레이션")
    lines.append("────────────")
    if req.qty is not None and req.unit_price is not None:
        lines.append(f"- 수량 : {req.qty:.0f} EA")
        lines.append(f"- 단가 : {fmt_money(req.unit_price, 'USD')}")
    if req.total_value_usd is not None:
        lines.append(f"- 과세가격(USD) : {fmt_money(req.total_value_usd, 'USD')}")
    if req.duty_usd is not None:
        lines.append(f"- 예상 관세(USD) : {fmt_money(req.duty_usd, 'USD')}")
    if req.usd_krw_rate and req.total_value_krw is not None and req.duty_krw is not None:
        lines.append("")
        lines.append(f"(환율 기준 : 1 USD = {req.usd_krw_rate:,.2f}원)")
        lines.append(f"- 과세가격(원화) : {fmt_money(req.total_value_krw, 'KRW')}")
        lines.append(f"- 예상 관세(원화) : {fmt_money(req.duty_krw, 'KRW')}")

    lines.append("")
    lines.append("※ 안내 드린 내용은 PoC 서비스에서 자동 계산된 값으로,")
    lines.append("   실제 신고·납부 금액과는 차이가 있을 수 있습니다.")
    lines.append("   최종 판단은 반드시 CBP 및 전문가 리뷰를 거쳐 주세요.")
    lines.append("")
    lines.append("궁금하신 점이 있으시면 회신으로 편하게 문의 주세요.")
    lines.append("")
    lines.append("감사합니다.")
    if req.query_prompt:
        lines.append("")
        lines.append(f"조회 Prompt : {req.query_prompt}")

    body = "\n".join(lines)
    return {"to": req.email_to, "subject": subject, "body": body}


@app.post("/api/send_email")
def send_email(req: EmailSendRequest, x_access_token: Optional[str] = Header(None)):
    _enforce_email_policy(req.email_to, x_access_token)
    try:
        send_email_via_gmail(req.email_to, req.subject, req.body)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /api/send_email")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
