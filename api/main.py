# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import time

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("AMIROBOTAI_MOCK", "false").lower() == "true"

STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SITE_URL = os.getenv("SITE_URL", "https://amirobotai.com")

stripe.api_key = STRIPE_SECRET

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://amirobotai.com,http://amirobotai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are amiRobotAI — an expert, practical, and clear AI companion for robotics, "
    "process automation, and industrial IoT. You are an AI system (not a human) and "
    "say so if asked. You speak in the user's language (detect from their message).\n\n"
    "YOUR EXPERTISE:\n"
    "- RPA (Robotic Process Automation): workflow design, bot architecture, tool "
    "selection (UiPath, Automation Anywhere, Power Automate, n8n, Zapier), process "
    "mining, exception handling, scaling strategies\n"
    "- Industrial robotics: robot types (SCARA, 6-axis, delta, cobot), kinematics "
    "basics, end-effector selection, cell layout, safety standards (ISO 10218, "
    "ISO/TS 15066), PLC integration\n"
    "- Process automation: workflow optimization, bottleneck analysis, ROI calculation, "
    "change management, KPI definition (cycle time, throughput, OEE)\n"
    "- IoT & edge computing: sensor selection, communication protocols (MQTT, OPC-UA, "
    "Modbus), edge vs cloud architecture, predictive maintenance, data pipelines\n"
    "- AI in automation: computer vision for QC, NLP for document processing, "
    "anomaly detection, reinforcement learning for control, digital twins\n"
    "- Integration: ERP/MES connectivity, API design for automation, middleware, "
    "data transformation, legacy system bridging\n\n"
    "OUTPUT FORMAT:\n"
    "- Use 🤖/⚙️/📊/✅/⚠️ icons for visual clarity\n"
    "- Structure advice with clear implementation steps\n"
    "- Include ROI considerations when relevant\n"
    "- Suggest specific tools/platforms when appropriate\n\n"
    "CRITICAL RULES:\n"
    "- Safety first: always highlight safety standards and risk assessment for "
    "physical robotics applications.\n"
    "- Never recommend bypassing safety interlocks or guards.\n"
    "- Be honest about automation limitations — not everything should be automated.\n"
    "- Recommend professional integrators for complex physical installations.\n"
    "- For RPA, warn about brittle bots and recommend resilient design patterns.\n"
    "- Do not fabricate specific ROI numbers — teach frameworks to calculate them."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None
    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.4, max_tokens=600, timeout=30, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.4, max_tokens=600, timeout=30, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="amiRobotAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "EN"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


@app.get("/")
def root():
    return {"service": "amiRobotAI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "stripe_configured": bool(STRIPE_SECRET and STRIPE_PRICE_ID),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/robot/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many requests — please wait a moment")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "EN"

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp = crisis_response(locale.lower())
        return ChatResponse(
            response=resp, engine="safety-layer", decision="crisis-intercept",
            certified=True, signature=_sign(resp), is_crisis_response=True,
        )

    if ENGINE:
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply, engine=result.engine, decision=result.decision,
            certified=True, signature=_sign(result.reply), is_crisis_response=False,
            concordance=result.concordance, latency_s=result.latency_s,
        )

    mock = "Great question! In production, I'd cross-verify this with two AI engines. Check back soon for verified automation insights."
    return ChatResponse(
        response=mock, engine="mock", decision="mock",
        certified=False, signature=_sign(mock), is_crisis_response=False,
    )


@app.post("/create-checkout-session")
def create_checkout():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(503, "Payment system not configured yet")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=SITE_URL + "?payment=success",
            cancel_url=SITE_URL + "?payment=cancelled",
        )
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Payment error: {e.user_message or str(e)}")
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")
    if event["type"] == "checkout.session.completed":
        print(f"[STRIPE] New sub: {event['data']['object'].get('customer_email', '?')}")
    elif event["type"] == "customer.subscription.deleted":
        print(f"[STRIPE] Sub cancelled: {event['data']['object'].get('id')}")
    return {"received": True}


@app.post("/gdpr/data-request")
def gdpr_data():
    return {
        "message": "amiRobotAI does not store chat messages or automation queries. Processing is stateless. "
                   "For Stripe subscription data, email privacy@amirobotai.com.",
        "data_stored": "none",
    }


@app.delete("/gdpr/delete")
def gdpr_delete():
    return {
        "message": "No personal data held server-side. "
                   "For Stripe data deletion, email privacy@amirobotai.com.",
        "status": "no_data_held",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
