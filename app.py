"""
LeadFinder — Gemini-powered public-web lead finder for Vercel.

The user can type natural language such as:
    "Find 10 people to reach at Clay for marketing"
    "Find senior sales people at Stripe"
    "Who should I contact at Notion for partnerships?"

Gemini handles both intent understanding and web research through Google Search grounding.
The app never asks the user to provide a company URL just to begin a search.

Run locally:
    uvicorn app:app --reload

Vercel:
    app.py + requirements.txt
"""

import os
import re
import json
import asyncio
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types


app = FastAPI(title="LeadFinder")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

GEMINI = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_json(text: str) -> dict:
    """Parse Gemini's JSON even if it accidentally wraps it in ```json ... ```."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover the first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def object_to_dict(value: Any) -> Any:
    """Convert Google SDK/Pydantic objects to ordinary Python dictionaries."""
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except Exception:
            pass

    if isinstance(value, dict):
        return {k: object_to_dict(v) for k, v in value.items()}

    if isinstance(value, list):
        return [object_to_dict(v) for v in value]

    if hasattr(value, "__dict__"):
        try:
            return {
                k: object_to_dict(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    return value


def extract_sources(response: Any) -> list[dict]:
    """Extract URLs/titles from Gemini Google Search grounding metadata."""
    data = object_to_dict(response)
    sources = []

    candidates = data.get("candidates", []) if isinstance(data, dict) else []

    for candidate in candidates:
        metadata = candidate.get("grounding_metadata") or candidate.get(
            "groundingMetadata"
        ) or {}

        chunks = metadata.get("grounding_chunks") or metadata.get(
            "groundingChunks"
        ) or []

        for chunk in chunks:
            web = chunk.get("web") if isinstance(chunk, dict) else None
            if not web:
                continue

            uri = web.get("uri")
            title = web.get("title") or uri

            if uri and not any(s["url"] == uri for s in sources):
                sources.append({"title": title, "url": uri})

    return sources


# ---------------------------------------------------------------------------
# Gemini research
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the research engine for a lead-finding app.

Understand the user's natural-language request and SEARCH THE PUBLIC WEB yourself
using Google Search grounding.

CORE RULES:
- Never ask for a company URL or website.
- Resolve the company and official domain yourself.
- Find real people matching the requested function, department, seniority,
  location and other constraints.
- Prefer current employees and current roles.
- Prefer official company pages, team/leadership pages, biographies and
  reputable professional sources. LinkedIn and other public professional
  profiles are allowed when found through search.
- The requested number is a TARGET, not a requirement. Return fewer people
  if strong matches are unavailable. NEVER invent people.
- For "people to reach", prioritize relevant decision-makers and functional owners.
- Every person must have a real source supporting their identity/role.

EMAIL RESEARCH:
1. Search for the person's exact public professional email using their name,
   company, domain, title, public profiles, interviews, conferences, PDFs,
   GitHub and other reputable public sources.
2. If an exact email is found, return it as:
   email_status = "public"
   and include the source URL.
3. If no exact email is found, investigate public emails from other employees
   to determine the company's email pattern.
4. If the pattern is sufficiently supported, generate possible emails for the
   target person and mark them:
   email_status = "inferred"
5. Use multiple public examples when possible.
6. Confidence:
   - high: multiple examples support the same pattern
   - medium: limited but reasonable evidence
   - low: weak/ambiguous evidence
7. If there is not enough evidence, return an empty possible_emails list.
8. Never fabricate, verify, or claim deliverability for an inferred email.
9. Never use SMTP mailbox enumeration or MX records as proof of deliverability.
10. Never call an inferred email "verified".
11. Return up to 3 possible emails only when supported by evidence. Never create
    random alternatives.

SOURCE RULES:
- Use only URLs actually found during web research.
- Do not create or guess URLs.
- source_url must support the person's identity/role.
- Public email addresses must have a source supporting the exact address.
- Inferred emails must include the evidence/reason for the inference.

RETURN JSON ONLY:

{
  "company": "resolved company name",
  "domain": "official domain or null",
  "interpreted_request": "short description of the request",
  "people": [
    {
      "name": "Full Name",
      "title": "Current job title",
      "department": "Marketing/Sales/etc or null",
      "location": "Location or null",
      "email": "exact public email or null",
      "email_status": "public | inferred | not_found",
      "possible_emails": [
        {
          "email": "possible address",
          "type": "inferred",
          "confidence": "high | medium | low",
          "reason": "why this address is inferred"
        }
      ],
      "email_pattern": {
        "pattern": "{first}.{last}@company.com",
        "confidence": "high | medium | low",
        "evidence": ["public evidence supporting the pattern"]
      },
      "profile_url": "actual public profile URL or null",
      "source_url": "strongest actual source URL",
      "reason": "why this person matches"
    }
  ],
  "notes": "brief search coverage or limitations"
}

Search the public web now. Return the strongest real matches you can find.
Fewer strong matches are better than invented matches.
Do not put search-result snippets or made-up URLs into source_url.
Only use URLs actually present in the web research.
"""


async def gemini_research(user_message: str) -> tuple[dict, list[dict]]:
    if not GEMINI:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it in Vercel Environment Variables."
        )

    prompt = f"""
{SYSTEM_PROMPT}

USER REQUEST:
{user_message}

Search the public web now. Find the best matching real people.
Remember: fewer strong matches is better than invented matches.
"""

    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=5000,
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ],
    )

    def call():
        return GEMINI.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

    response = await asyncio.to_thread(call)
    sources = extract_sources(response)

    data = clean_json(response.text)

    if not isinstance(data, dict):
        raise ValueError("Gemini returned an unexpected response.")

    people = data.get("people")
    if not isinstance(people, list):
        data["people"] = []

    # Defensive cleanup: make sure a malformed model response cannot break UI.
    cleaned = []
    for p in data.get("people", []):
        if not isinstance(p, dict):
            continue

        name = str(p.get("name") or "").strip()
        title = str(p.get("title") or "").strip()

        if not name:
            continue

        status = str(p.get("email_status") or "not_found").lower()
        if status not in {"public", "inferred", "not_found"}:
            status = "not_found"

        email = p.get("email")
        if email is not None:
            email = str(email).strip() or None

        # If Gemini gives no email, status must be not_found.
        if not email:
            status = "not_found"

        cleaned.append(
            {
                "name": name,
                "title": title or "Role not found",
                "department": p.get("department"),
                "location": p.get("location"),
                "email": email,
                "email_status": status,
                "profile_url": p.get("profile_url"),
                "source_url": p.get("source_url"),
                "reason": p.get("reason") or "",
            }
        )

    data["people"] = cleaned

    return data, sources


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatIn(BaseModel):
    message: str


@app.get("/api/health")
async def health():
    return {
        "ok": bool(GEMINI),
        "gemini_configured": bool(GEMINI),
        "model": GEMINI_MODEL,
        "search": "Google Search grounding",
    }


@app.post("/api/chat")
async def chat(body: ChatIn):
    message = (body.message or "").strip()

    if not message:
        return {
            "steps": ["Please enter what kind of person you want to find."],
            "results": [],
            "reply": "Tell me who you want to reach and which company or function.",
        }

    steps = [
        "Understanding your request…",
        "Gemini is searching the public web…",
    ]

    try:
        data, sources = await gemini_research(message)
    except Exception as exc:
        # Don't expose API keys or internal details to the browser.
        print("Gemini research error:", repr(exc))
        return {
            "steps": steps + [
                "Gemini could not complete the web research."
            ],
            "results": [],
            "reply": (
                "I couldn't complete the search right now. "
                "Check that GEMINI_API_KEY is configured and that the selected "
                "Gemini model supports Google Search grounding."
            ),
        }

    people = data.get("people", [])
    requested_company = data.get("company") or "the requested company"
    domain = data.get("domain")

    steps.append(
        f"Found {len(people)} relevant public contacts"
        + (f" at {requested_company}" if requested_company else "")
    )

    public_emails = sum(
        1 for p in people if p.get("email_status") == "public"
    )
    inferred_emails = sum(
        1 for p in people if p.get("email_status") == "inferred"
    )

    if people:
        reply = (
            f"I found {len(people)} relevant people for {requested_company}."
        )

        if public_emails:
            reply += f" {public_emails} have publicly listed emails."
        if inferred_emails:
            reply += f" {inferred_emails} have inferred emails."
        if domain:
            reply += f" Company domain: {domain}."

        notes = data.get("notes")
        if notes:
            reply += f" {notes}"
    else:
        reply = (
            f"I couldn't find a strong public match for {requested_company}. "
            "I did not invent people just to reach the requested number."
        )

    return {
        "steps": steps,
        "results": people,
        "reply": reply,
        "company": requested_company,
        "domain": domain,
        "interpreted_request": data.get("interpreted_request"),
        "sources": sources[:20],
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)
