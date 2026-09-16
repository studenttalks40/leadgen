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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

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
You are the research engine inside a lead-finding application.

Your job is to understand the user's natural-language request and then SEARCH
THE PUBLIC WEB yourself using Google Search grounding.

IMPORTANT:
- Do NOT ask the user for a company URL.
- Do NOT tell the user to provide a website.
- Resolve the company from the company name in the request.
- Search the web for the company and the type of people the user wants.
- Return the most relevant real people you can find.
- The requested count is a TARGET, not a requirement. If only 4 strong matches
  are found when the user asked for 10, return 4. Never invent people to reach
  the requested count.
- Prefer current employees and current roles.
- Prefer first-party company pages, official team pages, company leadership pages,
  official biographies, and reputable professional/public sources.
- Use LinkedIn/public professional pages when they appear in Google Search.
- Match the requested department, function, seniority, location and other
  constraints as closely as the evidence allows.
- If the user says "marketing", include marketing/growth/brand/demand-generation
  people where appropriate, but do not return unrelated functions just to fill
  the count.
- If the user asks for "people to reach", prioritize people who are plausible
  business contacts for the stated goal, such as decision makers or functional
  owners.

EMAIL RULES:
- NEVER invent an email address and call it verified.
- Only put an email in `email` when the exact address is publicly shown by a
  source you found.
- If an email is not publicly found, set `email` to null and
  `email_status` to "not_found".
- If you derive an address from a clearly established company pattern, you may
  include it ONLY with `email_status` = "inferred" and it must never be described
  as verified.
- Never claim an email is deliverable merely because the company has MX records.
EMAIL DISCOVERY AND INFERENCE TASK

For every person you identify, try to find their professional work email using public web sources.

Follow this process in order:

STEP 1 — FIND AN EXACT PUBLIC EMAIL

Search the web for the person's exact professional email address.

Search using combinations of:
- person's full name
- company name
- company domain
- person's job title
- official company pages
- public speaker/conference pages
- public interviews
- public PDFs/documents
- GitHub or other professional profiles
- reputable business directories
- publicly accessible professional profiles

If you find an exact email address that is publicly associated with that person:

email_status = "public"

Return the exact email and the URL/source where it was found.

DO NOT call an email public unless the exact address was actually found in a public source.

--------------------------------------------------

STEP 2 — IF NO EXACT EMAIL IS FOUND

If you cannot find an exact public email for the person, DO NOT stop.

Investigate the company's email naming convention.

Look for publicly available email addresses belonging to OTHER employees at the same company.

For example, if you find:

john.smith@company.com
sarah.jones@company.com
mike.brown@company.com

you may determine that the company appears to use:

{first}.{last}@company.com

Use multiple examples whenever possible rather than relying on a single example.

--------------------------------------------------

STEP 3 — GENERATE POSSIBLE EMAILS

If there is sufficient evidence for a company email pattern, generate possible email addresses for the target person.

Example:

Person:
Jane Doe

Company:
Acme

Observed company pattern:
{first}.{last}@acme.com

Possible email:
jane.doe@acme.com

Return:

{
  "email": null,
  "email_status": "not_found",
  "possible_emails": [
    {
      "email": "jane.doe@acme.com",
      "type": "inferred",
      "confidence": "high",
      "reason": "The company appears to use the firstname.lastname format based on publicly available employee emails."
    }
  ]
}

--------------------------------------------------

STEP 4 — CONFIDENCE

Assign confidence based on evidence.

HIGH:
- Multiple public employee emails support the same pattern.
- The target person's full name and company domain are known.
- The inferred address follows the observed pattern exactly.

MEDIUM:
- The pattern is supported by limited public evidence.
- There is some uncertainty about the company's naming convention.

LOW:
- The pattern is weakly supported or only one ambiguous example exists.

If there is not enough evidence to infer an email, return:

"possible_emails": []

Do NOT invent an email simply because it looks plausible.

--------------------------------------------------

STEP 5 — MULTIPLE POSSIBLE EMAILS

If several company patterns are supported by evidence, you may return up to 3 possible emails.

Example:

{
  "possible_emails": [
    {
      "email": "jane.doe@company.com",
      "type": "inferred",
      "confidence": "high",
      "reason": "Matches the most frequently observed company pattern."
    },
    {
      "email": "jdoe@company.com",
      "type": "inferred",
      "confidence": "medium",
      "reason": "Matches a secondary company email pattern found in public sources."
    }
  ]
}

Never generate random alternatives.

--------------------------------------------------

IMPORTANT RULES

1. Never fabricate a public email.
2. Never claim an inferred email is verified.
3. Never claim an inferred email is deliverable.
4. Clearly distinguish PUBLIC from INFERRED.
5. Use web evidence whenever possible.
6. Prefer official company sources and reputable public sources.
7. Do not use SMTP mailbox enumeration.
8. Do not ask the user for the company's website or URL.
9. Resolve the company and domain yourself using web search.
10. If the requested number of people cannot be found, return the strongest verified matches instead of inventing people.
11. The requested count is a target, not a requirement.
12. Every person must have a source supporting their identity/role.
13. Every public email should have a source supporting the exact email.
14. Inferred emails must include the evidence/reason for the inference.

RETURN FORMAT

Return JSON only:

{
  "people": [
    {
      "name": "",
      "title": "",
      "department": "",
      "company": "",
      "domain": "",

      "email": "",
      "email_status": "public|not_found",

      "possible_emails": [
        {
          "email": "",
          "type": "inferred",
          "confidence": "high|medium|low",
          "reason": ""
        }
      ],

      "email_pattern": {
        "pattern": "",
        "confidence": "high|medium|low",
        "evidence": []
      },

      "profile_url": "",
      "source_url": "",
      "reason": ""
    }
  ]
}
OUTPUT we want :
just the Name of person , where you can find that person like link to x or linkedIn and email if aviable if you did not find email just give all the possible cobination or emial the person an hae as you already known the company email 
RETURN FORMAT

Return JSON only:

{
  "people": [
    {
      "name": "",
      "title": "",
      "department": "",
      "company": "",
      "domain": "",

      "email": "",
      "email_status": "public|not_found",

      "possible_emails": [
        {
          "email": "",
          "type": "inferred",
          "confidence": "high|medium|low",
          "reason": ""
        }
      ],

      "email_pattern": {
        "pattern": "",
        "confidence": "high|medium|low",
        "evidence": []
      },

      "profile_url": "",
      "source_url": "",
      "reason": ""
    }
  ]
}
Use this exact shape:

{
  "company": "resolved company name",
  "domain": "official domain or null",
  "interpreted_request": "short description of what the user wants",
  "people": [
    {
      "name": "Full Name",
      "title": "Current job title",
      "department": "Marketing/Sales/etc or null",
      "location": "Location or null",
      "email": "exact public email or null",
      "email_status": "public | inferred | not_found",
      "profile_url": "public professional/profile URL if found, otherwise null",
      "source_url": "strongest source URL supporting this person",
      "reason": "one short sentence explaining why this person matches"
    }
  ],
  "notes": "brief note about search coverage or limitations"
}

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
