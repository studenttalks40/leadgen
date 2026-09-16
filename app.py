"""
LeadFinder AI
Generic natural-language company/contact researcher.

Run locally:
    uvicorn app:app --reload

Environment:
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=openrouter/free
    APP_URL=https://your-app.vercel.app
"""

import os
import json
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# APP CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="LeadFinder AI",
    version="2.0.0"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Generic free router.
# You can override this with another OpenRouter model later.
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "liquid/lfm-2.5-2.6b:free"
).strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

APP_URL = os.getenv(
    "APP_URL",
    "http://localhost:8000"
).strip()


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatIn(BaseModel):
    message: str


# ============================================================
# DEFAULT EMPTY RESPONSE
# ============================================================

def empty_response(message: str = "") -> dict:
    return {
        "company": "",
        "domain": "",
        "interpreted_request": {
            "count": 10,
            "department": "",
            "role": "",
            "seniority": "",
            "location": ""
        },
        "email_pattern": {
            "pattern": "",
            "confidence": "",
            "evidence": []
        },
        "people": [],
        "results": [],
        "notes": [],
        "reply": message,
        "count": 0
    }


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json(text: str) -> Any:
    """
    Extract JSON from model output.

    Handles:
    - normal JSON
    - ```json ... ```
    - extra text surrounding JSON
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # First try the entire response.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Find JSON array as fallback.
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def normalize_status(value: Any) -> str:
    status = safe_string(value).lower()

    if status in {
        "public",
        "publicly listed",
        "public_email",
        "published"
    }:
        return "public"

    if status in {
        "inferred",
        "inference",
        "pattern",
        "probable"
    }:
        return "inferred"

    if status in {
        "invalid",
        "rejected"
    }:
        return "invalid"

    return "not_found"


def normalize_person(person: Any) -> dict | None:
    """
    Convert whatever the model returns into the exact structure
    expected by the frontend.
    """

    if not isinstance(person, dict):
        return None

    name = safe_string(
        person.get("name")
        or person.get("full_name")
    )

    if not name:
        return None

    role = safe_string(
        person.get("role")
        or person.get("title")
        or person.get("job_title")
    )

    department = safe_string(
        person.get("department")
    )

    seniority = safe_string(
        person.get("seniority")
    )

    location = safe_string(
        person.get("location")
    )

    email = safe_string(
        person.get("email")
    )

    email_status = normalize_status(
        person.get("email_status")
        or person.get("status")
    )

    profile_url = safe_string(
        person.get("profile_url")
        or person.get("linkedin_url")
        or person.get("profile")
    )

    source_url = safe_string(
        person.get("source_url")
        or person.get("source")
    )

    reason = safe_string(
        person.get("reason")
        or person.get("email_reason")
    )

    possible_emails = person.get("possible_emails", [])

    if not isinstance(possible_emails, list):
        possible_emails = []

    possible_emails = [
        safe_string(x)
        for x in possible_emails
        if safe_string(x)
    ][:5]

    # If an email exists but the model forgot to specify the status,
    # treat it as not_found only if it is empty; otherwise infer status.
    if email and email_status == "not_found":
        email_status = "public"

    return {
        "name": name,
        "role": role,
        "department": department,
        "seniority": seniority,
        "location": location,
        "email": email,
        "email_status": email_status,
        "possible_emails": possible_emails,
        "profile_url": profile_url,
        "source_url": source_url,
        "reason": reason
    }


def normalize_result(data: Any) -> dict:
    """
    Normalize the AI response so the frontend always receives:

        company
        domain
        interpreted_request
        email_pattern
        people
        results
        notes
        reply
        count
    """

    if not isinstance(data, dict):
        return empty_response(
            "I couldn't structure the research results."
        )

    company = safe_string(
        data.get("company")
    )

    domain = safe_string(
        data.get("domain")
    )

    interpreted = data.get(
        "interpreted_request",
        {}
    )

    if not isinstance(interpreted, dict):
        interpreted = {}

    count = interpreted.get("count", 10)

    try:
        count = int(count)
    except Exception:
        count = 10

    count = max(1, min(count, 50))

    interpreted_request = {
        "count": count,
        "department": safe_string(
            interpreted.get("department")
        ),
        "role": safe_string(
            interpreted.get("role")
        ),
        "seniority": safe_string(
            interpreted.get("seniority")
        ),
        "location": safe_string(
            interpreted.get("location")
        )
    }

    email_pattern = data.get(
        "email_pattern",
        {}
    )

    if not isinstance(email_pattern, dict):
        email_pattern = {}

    pattern = {
        "pattern": safe_string(
            email_pattern.get("pattern")
        ),
        "confidence": safe_string(
            email_pattern.get("confidence")
        ),
        "evidence": email_pattern.get(
            "evidence",
            []
        )
    }

    if not isinstance(pattern["evidence"], list):
        pattern["evidence"] = []

    pattern["evidence"] = [
        safe_string(x)
        for x in pattern["evidence"]
        if safe_string(x)
    ][:10]

    # The model may use either "people" or "results".
    raw_people = data.get("people")

    if not isinstance(raw_people, list):
        raw_people = data.get("results", [])

    if not isinstance(raw_people, list):
        raw_people = []

    people = []

    seen = set()

    for raw_person in raw_people:

        person = normalize_person(raw_person)

        if not person:
            continue

        key = (
            person["name"].lower(),
            person["role"].lower()
        )

        if key in seen:
            continue

        seen.add(key)
        people.append(person)

    # IMPORTANT:
    # Contacts are people found by research, not people whose
    # emails happened to pass a separate verification step.
    people = people[:count]

    notes = data.get("notes", [])

    if not isinstance(notes, list):
        notes = [notes]

    notes = [
        safe_string(x)
        for x in notes
        if safe_string(x)
    ][:10]

    reply = safe_string(
        data.get("reply")
    )

    # Generate a clean fallback summary.
    if not reply:

        if people:
            reply = (
                f"Found {len(people)} relevant "
                f"public contact"
                f"{'s' if len(people) != 1 else ''}"
                f"{' at ' + company if company else ''}."
            )
        else:
            reply = (
                f"No matching public contacts were found"
                f"{' for ' + company if company else ''}."
            )

    return {
        "company": company,
        "domain": domain,
        "interpreted_request": interpreted_request,
        "email_pattern": pattern,
        "people": people,

        # Keep both names for frontend compatibility.
        "results": people,

        "notes": notes,
        "reply": reply,

        # THIS is the number the UI should display.
        "count": len(people)
    }


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = r"""
You are LeadFinder AI, a business-contact research assistant.

Your job is to understand natural-language requests and research
PUBLIC information on the web.

Users may say things like:

"find me 10 people to contact at Stripe in marketing"

"who should I reach out to at OpenAI for partnerships?"

"find sales leaders at Notion"

"give me 5 founders at a fintech company"

"find someone in HR at Canva in the US"

The request can contain:
- company
- number of people
- department
- job function
- exact role
- seniority
- location
- other relevant constraints

You MUST understand the request semantically.
Do not rely on a rigid regex parser.

============================================================
RESEARCH
============================================================

Use web search to find real people who are publicly associated
with the requested company.

Prefer:
1. Official company websites
2. Official leadership/team pages
3. Company newsroom pages
4. Public professional profiles
5. Reputable public business sources
6. Public interviews, conference pages, podcasts, articles,
   company announcements, etc.

Do NOT invent people.

A person should only be returned when there is reasonable
public evidence that they work at or are associated with the
requested company.

============================================================
RELEVANCE
============================================================

Match the user's request.

For example, if the user asks for:

"marketing"

include people whose actual role relates to marketing,
growth marketing, product marketing, brand marketing,
demand generation, marketing leadership, etc.

If the user asks for:

"sales"

prioritize sales leadership, sales operations, revenue,
business development, account executives, etc.

If the user asks for a specific role, prioritize that role.

If the user specifies seniority, respect it.

If the user specifies a location, use it when public evidence
supports it.

Do not simply return random executives from the company.

============================================================
EMAILS
============================================================

Emails require special care.

FIRST:
Look for an exact individual email address that is publicly
documented on a public source.

If an exact email is publicly documented:

    email_status = "public"

If no exact public email exists:

You MAY infer an email address only when there is public
evidence for the company's email naming convention.

For example, if public employee emails demonstrate:

firstname.lastname@company.com

and you find:

John Smith

you may infer:

john.smith@company.com

But mark it:

    email_status = "inferred"

and explain the evidence.

NEVER claim an inferred email is verified.

NEVER invent a supposedly verified email.

NEVER use SMTP mailbox enumeration.

NEVER use Gravatar as proof that an email mailbox exists.

If there is not enough evidence to infer an email:

    email = ""
    email_status = "not_found"

The person should STILL be returned.

A missing email does NOT mean the person should be removed.

============================================================
SOURCES
============================================================

For every person, provide:
- profile_url when available
- source_url when available
- a short reason explaining why the person matches

Do not fabricate URLs.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
  "company": "",
  "domain": "",
  "interpreted_request": {
    "count": 10,
    "department": "",
    "role": "",
    "seniority": "",
    "location": ""
  },
  "email_pattern": {
    "pattern": "",
    "confidence": "",
    "evidence": []
  },
  "people": [
    {
      "name": "",
      "role": "",
      "department": "",
      "seniority": "",
      "location": "",
      "email": "",
      "email_status": "public",
      "possible_emails": [],
      "profile_url": "",
      "source_url": "",
      "reason": ""
    }
  ],
  "notes": [],
  "reply": ""
}

============================================================
IMPORTANT
============================================================

Return the PEOPLE you actually found.

Do NOT put the people only inside "reply".

"reply" should be a short human-readable summary.

If you found 4 people, people must contain 4 people.

If you found 0 people, people must be [].

The requested count is a maximum, not a requirement to invent
people.

Never invent a person merely to reach the requested count.

Keep results concise and useful.
"""


# ============================================================
# OPENROUTER RESEARCH
# ============================================================

async def openrouter_research(
    user_message: str
) -> dict:

    if not OPENROUTER_API_KEY:
        return empty_response(
            "OPENROUTER_API_KEY is not configured."
        )

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": "LeadFinder AI"
    }

    user_prompt = f"""
Research this business-contact request:

{user_message}

Use public web search.

Understand the company, requested department/role,
seniority, location, and requested number of contacts.

Return structured JSON exactly according to the system
instructions.
"""

    payload = {
        "model": OPENROUTER_MODEL,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        # FAST ONE-SHOT WEB SEARCH.
        # This is intentionally the web plugin, not the newer
        # agentic server tool. The server tool can search 0..N times
        # and is slower for this lead-finder workflow.
        "plugins": [
            {
                "id": "web",
                "max_results": 3,
                "search_prompt": (
                    "Find public sources relevant to the user's company/contact request. "
                    "Prioritize official company team/leadership pages, public professional profiles, "
                    "company announcements, and other trustworthy public sources. "
                    "Focus on identifying real people and their current roles. "
                    "Do not invent people or emails."
                )
            }
        ],

        "temperature": 0,

        # Keep the final response small and fast.
        "max_tokens": 1800,

        # Ask the model for machine-readable JSON.
        "response_format": {
            "type": "json_object"
        }
    }

    try:

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10,
                read=45,
                write=15,
                pool=10
            )
        ) as client:

            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload
            )

        # ----------------------------------------------------
        # OpenRouter error
        # ----------------------------------------------------

        if response.status_code != 200:

            try:
                error_data = response.json()
            except Exception:
                error_data = response.text

            print(
                "OPENROUTER ERROR:",
                response.status_code,
                error_data
            )

            result = empty_response(
                f"Research service returned HTTP "
                f"{response.status_code}."
            )
            result["error"] = "openrouter_http_error"

            result["notes"] = [
                "OpenRouter request failed.",
                f"HTTP status: {response.status_code}"
            ]

            return result

        # ----------------------------------------------------
        # Parse response
        # ----------------------------------------------------

        data = response.json()

        print(
            "OPENROUTER MODEL:",
            data.get(
                "model",
                OPENROUTER_MODEL
            )
        )

        choices = data.get(
            "choices",
            []
        )

        if not choices:
            return empty_response(
                "The research service returned no result."
            )

        message = choices[0].get(
            "message",
            {}
        )

        content = message.get(
            "content",
            ""
        )

        # Some providers can return content as an array.
        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        parts.append(
                            safe_string(
                                item.get("text")
                            )
                        )

                elif isinstance(item, str):
                    parts.append(item)

            content = "\n".join(parts)

        content = safe_string(content)

        # ----------------------------------------------------
        # Parse structured JSON.
        # ----------------------------------------------------

        result = clean_json(content)

        if not isinstance(result, dict):

            print(
                "OPENROUTER INVALID JSON:",
                content[:2000]
            )

            return empty_response(
                "The research service returned an "
                "unexpected response format."
            )

        return normalize_result(result)

    except httpx.TimeoutException:

        print(
            "OPENROUTER TIMEOUT"
        )

        result = empty_response(
            "OpenRouter web search timed out after 45 seconds. "
            "The search request did not finish."
        )
        result["error"] = "openrouter_timeout"
        return result

    except httpx.RequestError as exc:

        print(
            "OPENROUTER REQUEST ERROR:",
            repr(exc)
        )

        result = empty_response(
            "Could not connect to the research service."
        )
        result["error"] = "openrouter_request_error"
        return result

    except Exception as exc:

        print(
            "OPENROUTER EXCEPTION:",
            repr(exc)
        )

        result = empty_response(
            "Something went wrong while researching."
        )
        result["error"] = "openrouter_exception"
        return result


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
async def chat(body: ChatIn):

    message = safe_string(
        body.message
    )

    if not message:
        result = empty_response(
            "Please enter a company/contact request."
        )

        result["steps"] = [
            "Waiting for a request..."
        ]

        return result

    steps = [
        "Understanding your request...",
        "Searching public sources...",
        "Matching relevant people..."
    ]

    result = await openrouter_research(
        message
    )

    # Add processing information for the UI.
    if result.get("error") == "openrouter_timeout":
        result["steps"] = steps + ["Research timed out."]
    elif result.get("error"):
        result["steps"] = steps + ["Research service error."]
    else:
        result["steps"] = steps + [
            f"Found {result.get('count', 0)} "
            f"relevant contact"
            f"{'s' if result.get('count', 0) != 1 else ''}."
        ]

    # --------------------------------------------------------
    # Make absolutely sure the count is tied to actual
    # structured contacts.
    # --------------------------------------------------------

    people = result.get(
        "people",
        []
    )

    if not isinstance(people, list):
        people = []

    result["people"] = people
    result["results"] = people
    result["count"] = len(people)

    # --------------------------------------------------------
    # If the model somehow returned contacts but an empty
    # reply, create one.
    # --------------------------------------------------------

    if not result.get("reply"):

        company = result.get(
            "company",
            ""
        )

        if people:

            result["reply"] = (
                f"Found {len(people)} relevant "
                f"public contact"
                f"{'s' if len(people) != 1 else ''}"
                f"{' at ' + company if company else ''}."
            )

        else:

            result["reply"] = (
                "No matching public contacts were found."
            )

    return result


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "ok": True,
        "service": "LeadFinder AI",
        "openrouter_configured": bool(
            OPENROUTER_API_KEY
        ),
        "model": OPENROUTER_MODEL
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

@app.get("/")
async def index():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        return {
            "ok": True,
            "message": "LeadFinder API is running.",
            "frontend": "static/index.html not found"
        }

    return FileResponse(
        index_file
    )


if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static"
    )
