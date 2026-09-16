"""
LeadFinder — AI web-research lead finder

Run:
    uvicorn app:app --reload

Then open:
    http://localhost:8000
"""

import os
import json
import re
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

app = FastAPI(title="LeadFinder AI")


# ---------------------------------------------------------
# OPENROUTER
# ---------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

APP_URL = os.getenv(
    "APP_URL",
    "https://your-domain.vercel.app"
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def clean_json(text: str) -> Any:
    """
    Convert model output into JSON even if the model
    wraps it inside ```json ... ```
    """

    if not text:
        return {}

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {
        "company": "",
        "domain": "",
        "people": [],
        "reply": text,
        "notes": []
    }


# ---------------------------------------------------------
# AI RESEARCH PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = r"""
You are LeadFinder, an AI research assistant that finds public business
contacts from the web.

The user will give you a natural-language request such as:

"find 10 people at Stripe in marketing"

"who should I contact at Clay for partnerships"

"find the head of marketing at OpenAI"

Your job:

1. Understand the request yourself.
2. Identify the company.
3. Identify the requested department, role, seniority, location,
   quantity and other constraints.
4. Use web search extensively.
5. Search the company's official website and other reliable public sources.
6. Find real people who actually work at the company.
7. Find a public business email when one is explicitly published.
8. If an email is not publicly available, investigate the company's
   public email convention using real publicly visible employee emails.
9. You may infer an email only when there is evidence supporting the pattern.
10. Never invent a person.
11. Never invent an email and call it verified.
12. Never claim an inferred email is deliverable.
13. Do not use SMTP mailbox enumeration.
14. Do not use Gravatar as mailbox verification.

Email statuses:

"public"
    Exact email was publicly found.

"inferred"
    Email was generated from a supported company email pattern.

"not_found"
    No reliable email could be found or inferred.

For inferred emails:
- confidence must be high, medium or low.
- explain the evidence.
- never call the email verified.

Return ONLY valid JSON.

Use this exact structure:

{
  "company": "Company Name",
  "domain": "company.com",
  "interpreted_request": {
    "count": 10,
    "department": "marketing",
    "role": "",
    "seniority": "",
    "location": ""
  },
  "email_pattern": {
    "pattern": "{first}.{last}@company.com",
    "confidence": "high",
    "evidence": [
      "Public employee email found using firstname.lastname pattern"
    ]
  },
  "people": [
    {
      "name": "Jane Smith",
      "role": "VP of Marketing",
      "department": "Marketing",
      "seniority": "VP",
      "location": "",
      "email": "jane.smith@company.com",
      "email_status": "public",
      "possible_emails": [],
      "profile_url": "",
      "source_url": "",
      "reason": "Publicly listed employee and email"
    },
    {
      "name": "John Doe",
      "role": "Head of Marketing",
      "department": "Marketing",
      "seniority": "Head",
      "location": "",
      "email": "",
      "email_status": "inferred",
      "possible_emails": [
        {
          "email": "john.doe@company.com",
          "type": "inferred",
          "confidence": "high",
          "reason": "Matches the firstname.lastname pattern found in public company emails"
        }
      ],
      "profile_url": "",
      "source_url": "",
      "reason": "Person publicly identified at company"
    }
  ],
  "notes": [
    "Only publicly supported information was included."
  ],
  "reply": "Short natural-language summary of what was found."
}

IMPORTANT:

- Respect the requested count when possible.
- If the user asks for 10 people, try to find 10.
- Do not fabricate people just to reach the requested count.
- Prefer official company pages, company leadership pages,
  conference pages, public articles, interviews and other credible sources.
- A LinkedIn URL may be included when publicly discoverable.
- Search for actual public emails before inferring anything.
- If public employee emails reveal a pattern, document that evidence.
- Generate at most 3 possible emails per person.
- Only generate possible emails when the pattern has evidence.
- If no evidence supports an email pattern, leave possible_emails empty.
- Keep the final reply short.
"""


# ---------------------------------------------------------
# OPENROUTER RESEARCH
# ---------------------------------------------------------

async def openrouter_research(user_message: str) -> dict:
    if not OPENROUTER_API_KEY:
        return {
            "company": "",
            "domain": "",
            "people": [],
            "notes": ["OPENROUTER_API_KEY is not configured."],
            "reply": "OpenRouter is not configured. Add OPENROUTER_API_KEY to your environment variables."
        }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL,
        "X-Title": "LeadFinder AI"
    }

    payload = {
        "model": OPENROUTER_MODEL,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        "tools": [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "auto",
                    "max_results": 5,
                    "max_total_results": 15
                }
            },
            {
                "type": "openrouter:web_fetch",
                "parameters": {
                    "engine": "openrouter",
                    "max_content_tokens": 20000
                }
            }
        ],

        "temperature": 0.1,
        "max_tokens": 5000
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:

            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                try:
                    error_data = response.json()
                except Exception:
                    error_data = response.text

                return {
                    "company": "",
                    "domain": "",
                    "people": [],
                    "notes": [
                        f"OpenRouter error: HTTP {response.status_code}",
                        str(error_data)
                    ],
                    "reply": "The AI research request failed."
                }

            data = response.json()

            choices = data.get("choices", [])

            if not choices:
                return {
                    "company": "",
                    "domain": "",
                    "people": [],
                    "notes": ["OpenRouter returned no choices."],
                    "reply": "No research result was returned."
                }

            message = choices[0].get("message", {})

            content = message.get("content", "")

            # Some providers may return structured content
            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)

                content = "\n".join(text_parts)

            result = clean_json(content)

            if not isinstance(result, dict):
                result = {
                    "company": "",
                    "domain": "",
                    "people": [],
                    "notes": [],
                    "reply": str(result)
                }

            return result

    except httpx.TimeoutException:
        return {
            "company": "",
            "domain": "",
            "people": [],
            "notes": ["OpenRouter request timed out."],
            "reply": "The research took too long and timed out. Please try again."
        }

    except Exception as e:
        return {
            "company": "",
            "domain": "",
            "people": [],
            "notes": [f"Research error: {str(e)}"],
            "reply": "Something went wrong while researching the request."
        }


# ---------------------------------------------------------
# NORMALIZE RESULTS
# ---------------------------------------------------------

def normalize_person(person: dict) -> dict:
    return {
        "name": person.get("name", ""),
        "role": person.get("role", ""),
        "department": person.get("department", ""),
        "seniority": person.get("seniority", ""),
        "location": person.get("location", ""),

        "email": person.get("email", ""),

        "email_status": person.get(
            "email_status",
            "not_found"
        ),

        "possible_emails": person.get(
            "possible_emails",
            []
        ),

        "email_pattern": person.get(
            "email_pattern",
            None
        ),

        "profile_url": person.get(
            "profile_url",
            ""
        ),

        "source_url": person.get(
            "source_url",
            ""
        ),

        "reason": person.get(
            "reason",
            ""
        )
    }


# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------

class ChatIn(BaseModel):
    message: str


# ---------------------------------------------------------
# CHAT API
# ---------------------------------------------------------

@app.post("/api/chat")
async def chat(body: ChatIn):

    message = body.message.strip()

    if not message:
        return {
            "steps": [],
            "results": [],
            "reply": "Tell me who you want to find."
        }

    steps = [
        "Understanding your request…",
        "Searching the public web…",
        "Finding relevant people…",
        "Checking public email evidence…"
    ]

    result = await openrouter_research(message)

    people = result.get("people", [])

    if not isinstance(people, list):
        people = []

    normalized_people = []

    for person in people:
        if isinstance(person, dict):
            normalized_people.append(
                normalize_person(person)
            )

    # Keep requested result count reasonable
    normalized_people = normalized_people[:25]

    result["people"] = normalized_people

    # Frontend compatibility
    result["results"] = normalized_people

    result["steps"] = steps + [
        f"Found {len(normalized_people)} supported people."
    ]

    if not result.get("reply"):
        company = result.get("company", "")

        if normalized_people:
            result["reply"] = (
                f"I found {len(normalized_people)} relevant "
                f"people for {company}."
            )
        else:
            result["reply"] = (
                "I couldn't find enough publicly supported contacts "
                "for that request."
            )

    return result


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "ok": True,
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "model": OPENROUTER_MODEL
    }


# ---------------------------------------------------------
# FRONTEND
# ---------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
