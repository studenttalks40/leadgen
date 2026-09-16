"""
LeadFinder — Gemini-powered public-web lead finder MVP

Run:
    pip install -r requirements.txt
    set GEMINI_API_KEY=your_key
    uvicorn app:app --reload

Then open:
    http://localhost:8000

Notes:
- Gemini is used for natural-language intent parsing and ranking.
- People are discovered from public company/team/leadership pages and public search results.
- Email addresses are only marked "verified" when an exact public email is found on a public source.
- Pattern-generated emails are explicitly marked "inferred" and are NOT mailbox-verified.
- No SMTP mailbox enumeration and no Gravatar verification.
"""

import os
import re
import json
import asyncio
from urllib.parse import urljoin, urlparse, quote_plus
from typing import Optional, Any

import httpx
import dns.resolver
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


app = FastAPI(title="LeadFinder")

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36 "
        "LeadFinder/1.0"
    )
}

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

GEMINI_CLIENT = None
if genai and GEMINI_KEY:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_KEY)

PATTERNS = [
    "{first}.{last}",
    "{f}{last}",
    "{first}{last}",
    "{first}",
    "{last}.{first}",
    "{first}_{last}",
]

BAD_PATH = re.compile(
    r"privacy|terms|login|signin|signup|cart|pricing|jobs|career|"
    r"contact|support|blog|press|news|cookie|legal",
    re.I,
)

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I
)

ROLE_RE = re.compile(
    r"\b(CEO|CFO|COO|CTO|CIO|CMO|CPO|CSO|"
    r"Chief|Founder|Co-Founder|President|Partner|"
    r"VP|Vice President|Head of|Director|Manager|Lead|"
    r"Marketing|Sales|Engineering|Product|Finance|"
    r"Operations|Legal|HR|People|Growth|Revenue)\b",
    re.I,
)

DEPARTMENT_ALIASES = {
    "marketing": [
        "marketing", "growth", "brand", "demand generation",
        "communications", "content", "performance marketing",
    ],
    "sales": [
        "sales", "revenue", "business development", "bd",
        "account executive", "commercial",
    ],
    "engineering": [
        "engineering", "software", "developer", "technical",
        "technology", "platform", "infrastructure",
    ],
    "product": [
        "product", "product management", "product manager",
    ],
    "finance": [
        "finance", "financial", "accounting", "treasury",
    ],
    "hr": [
        "human resources", "people", "talent", "hr",
        "recruiting", "recruitment",
    ],
    "operations": [
        "operations", "strategy", "business operations",
    ],
    "legal": [
        "legal", "compliance", "privacy", "counsel",
    ],
}

SENIORITY_TERMS = {
    "c_level": ["ceo", "cfo", "coo", "cto", "cmo", "cpo", "cio", "chief"],
    "vp": ["vp", "vice president"],
    "director": ["director"],
    "head": ["head of", "global head", "regional head"],
    "manager": ["manager"],
    "lead": ["lead", "principal"],
}


# ----------------------------- HTTP helpers -----------------------------

async def fetch(client: httpx.AsyncClient, url: str, timeout: float = 10) -> str:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            return r.text
    except Exception:
        pass
    return ""


def clean_domain(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/")[0]
    return value.strip(".")


async def find_domain(company: str) -> Optional[str]:
    """Resolve a company name using Clearbit autocomplete, then web search."""
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA) as c:
            r = await c.get(
                "https://autocomplete.clearbit.com/v1/companies/suggest",
                params={"query": company},
            )
            if r.status_code == 200:
                data = r.json()
                if data:
                    # Prefer an exact-ish name match.
                    company_low = company.lower().strip()
                    for item in data:
                        name = str(item.get("name", "")).lower()
                        domain = item.get("domain")
                        if domain and (company_low in name or name in company_low):
                            return clean_domain(domain)
                    for item in data:
                        if item.get("domain"):
                            return clean_domain(item["domain"])
    except Exception:
        pass

    # Fallback: search engine result URL.
    try:
        async with httpx.AsyncClient(timeout=10, headers=UA) as c:
            q = quote_plus(f"{company} official website")
            html = await fetch(c, f"https://html.duckduckgo.com/html/?q={q}")
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.select("a.result__a, a[href]"):
                href = a.get("href", "")
                if not href.startswith("http"):
                    continue
                d = clean_domain(urlparse(href).netloc)
                if d and "duckduckgo" not in d and "google" not in d:
                    return d
    except Exception:
        pass

    return None


async def mx_hosts(domain: str) -> list[str]:
    try:
        ans = await asyncio.to_thread(dns.resolver.resolve, domain, "MX")
        return sorted(str(r.exchange).rstrip(".") for r in ans)
    except Exception:
        return []


# ----------------------------- Public discovery -----------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def same_domain(url: str, domain: str) -> bool:
    try:
        host = urlparse(url).netloc.lower().split(":")[0]
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False


def canonical_url(url: str, domain: str) -> Optional[str]:
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        url = "https://" + domain + url
    if not url.startswith("http"):
        return None
    if not same_domain(url, domain):
        return None
    return url.split("#")[0]


def likely_people_link(href: str, text: str = "") -> bool:
    blob = f"{href} {text}"
    return bool(
        re.search(
            r"team|leadership|people|about|management|executives|"
            r"company|founders|our-team|meet-the-team",
            blob,
            re.I,
        )
    ) and not BAD_PATH.search(blob)


def extract_jsonld_people(soup: BeautifulSoup, source: str) -> list[dict]:
    people = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        try:
            data = json.loads(raw)
        except Exception:
            continue

        nodes = []
        if isinstance(data, list):
            nodes.extend(data)
        elif isinstance(data, dict):
            nodes.append(data)
            graph = data.get("@graph")
            if isinstance(graph, list):
                nodes.extend(graph)

        for n in nodes:
            if not isinstance(n, dict):
                continue
            typ = n.get("@type")
            if typ == "Person" or (isinstance(typ, list) and "Person" in typ):
                name = normalize_space(str(n.get("name", "")))
                role = normalize_space(str(n.get("jobTitle", "")))
                if name:
                    people.append({
                        "name": name,
                        "role": role,
                        "src": source,
                        "source_type": "company_page",
                        "email": None,
                    })
    return people


def extract_people_from_text(text: str, source: str) -> list[dict]:
    people = []

    # Common visible formats:
    # Jane Doe — VP Marketing
    # Jane Doe, Chief Marketing Officer
    # Jane Doe | Head of Marketing
    pattern = re.compile(
        r"\b([A-Z][A-Za-z'’\-]{1,25}(?:\s+[A-Z][A-Za-z'’\-]{1,25}){1,3})"
        r"\s*(?:—|–|\||:|,\s*)\s*"
        r"((?:Chief|Co[- ]Founder|Founder|President|VP|Vice President|"
        r"Head of|Director|Manager|Lead|Principal|Senior|Marketing|"
        r"Sales|Engineering|Product|Finance|Operations|Legal|HR)"
        r"[^.\n]{2,70})",
        re.I,
    )

    for m in pattern.finditer(text):
        name = normalize_space(m.group(1))
        role = normalize_space(m.group(2))
        if len(name.split()) >= 2 and ROLE_RE.search(role):
            people.append({
                "name": name,
                "role": role,
                "src": source,
                "source_type": "company_page",
                "email": None,
            })

    return people


async def scrape_people(domain: str) -> list[dict]:
    found: list[dict] = []
    seen_people = set()
    visited = set()

    async with httpx.AsyncClient(timeout=12, headers=UA) as c:
        home_url = f"https://{domain}"
        home = await fetch(c, home_url)
        if not home:
            return []

        soup = BeautifulSoup(home, "html.parser")
        urls = [home_url]

        for a in soup.find_all("a", href=True):
            href = canonical_url(a.get("href", ""), domain)
            text = normalize_space(a.get_text(" ", strip=True))
            if href and likely_people_link(href, text):
                if href not in urls:
                    urls.append(href)

        # Probe conventional pages too.
        for path in [
            "/about", "/team", "/leadership", "/company",
            "/people", "/about-us", "/management", "/founders",
        ]:
            urls.append(f"https://{domain}{path}")

        # Keep discovery bounded.
        urls = list(dict.fromkeys(urls))[:12]

        for url in urls:
            if url in visited:
                continue
            visited.add(url)

            html = home if url == home_url else await fetch(c, url)
            if not html:
                continue

            page_soup = BeautifulSoup(html, "html.parser")
            page_text = normalize_space(page_soup.get_text(" ", strip=True))[:120000]

            for p in extract_jsonld_people(page_soup, url):
                key = (p["name"].lower(), p["role"].lower())
                if key not in seen_people:
                    seen_people.add(key)
                    found.append(p)

            for p in extract_people_from_text(page_text, url):
                key = (p["name"].lower(), p["role"].lower())
                if key not in seen_people:
                    seen_people.add(key)
                    found.append(p)

            # Associate exact public emails with nearby names where possible.
            emails = [e.lower() for e in EMAIL_RE.findall(page_text)]
            for p in found:
                if p["src"] != url or p.get("email"):
                    continue
                first = p["name"].split()[0].lower()
                last = p["name"].split()[-1].lower()
                for email in emails:
                    local = email.split("@")[0]
                    if first in local and (last in local or first == local):
                        p["email"] = email
                        p["source_type"] = "public_email"
                        break

            if len(found) >= 60:
                break

    return found


async def public_search_people(company: str, domain: str, intent: dict) -> list[dict]:
    """Use public search results as a discovery fallback, not as an email database."""
    queries = [
        f'site:{domain} "team" "{company}"',
        f'site:{domain} "leadership" "{company}"',
        f'site:{domain} "VP" "{company}"',
        f'site:{domain} "Head of" "{company}"',
    ]

    if intent.get("department"):
        queries.insert(
            0, f'site:{domain} "{intent["department"]}" "{company}"'
        )

    out = []
    seen = set()

    async with httpx.AsyncClient(timeout=12, headers=UA) as c:
        for q in queries[:5]:
            try:
                html = await fetch(
                    c,
                    f"https://html.duckduckgo.com/html/?q={quote_plus(q)}",
                )
                soup = BeautifulSoup(html, "html.parser")
                for result in soup.select(".result")[:10]:
                    title = normalize_space(
                        (result.select_one(".result__title") or result).get_text(
                            " ", strip=True
                        )
                    )
                    snippet_node = result.select_one(".result__snippet")
                    snippet = normalize_space(
                        snippet_node.get_text(" ", strip=True)
                        if snippet_node else ""
                    )
                    link_node = result.select_one("a.result__a")
                    src = link_node.get("href", "") if link_node else ""
                    blob = f"{title} {snippet}"

                    # Extract plausible "Name — Role" from title/snippet.
                    for p in extract_people_from_text(blob, src or q):
                        key = (p["name"].lower(), p["role"].lower())
                        if key not in seen:
                            seen.add(key)
                            out.append(p)
            except Exception:
                continue

    return out


# ----------------------------- Intent -----------------------------

def local_parse_intent(msg: str) -> dict:
    text = normalize_space(msg)

    count_match = re.search(r"\b(\d{1,3})\b", text)
    count = int(count_match.group(1)) if count_match else 10
    count = max(1, min(count, 50))

    department = None
    for d in DEPARTMENT_ALIASES:
        if re.search(rf"\b{re.escape(d)}\b", text, re.I):
            department = d
            break

    if not department:
        for d, aliases in DEPARTMENT_ALIASES.items():
            if any(re.search(rf"\b{re.escape(a)}\b", text, re.I) for a in aliases):
                department = d
                break

    seniority = None
    for level, terms in SENIORITY_TERMS.items():
        if any(t in text.lower() for t in terms):
            seniority = level
            break

    # Company usually follows "at/for/from/of" and precedes optional department.
    company = ""
    m = re.search(
        r"\b(?:at|from|for|of)\s+([A-Za-z0-9&.,'’\- ]{2,80}?)(?:\s+(?:in|from|within)\s+"
        r"(?:marketing|sales|engineering|product|finance|hr|operations|legal)\b|$)",
        text,
        re.I,
    )
    if m:
        company = normalize_space(m.group(1)).strip(" .,")
    else:
        # e.g. "find 10 people at Stripe in marketing"
        m = re.search(
            r"\bat\s+([A-Za-z0-9&.,'’\- ]+?)(?:\s+in\s+.+)?$",
            text,
            re.I,
        )
        if m:
            company = normalize_space(m.group(1)).strip(" .,")

    if not company:
        company = text
        company = re.sub(
            r"^(find|get|show|give|list)\s+\d*\s*(people|contacts|leads|employees)?\s*",
            "",
            company,
            flags=re.I,
        )
        company = re.sub(
            r"\b(in|from|within)\s+(marketing|sales|engineering|product|finance|hr|operations|legal)\b.*$",
            "",
            company,
            flags=re.I,
        ).strip(" .,")

    role_keywords = []
    if department:
        role_keywords.extend(DEPARTMENT_ALIASES[department])

    return {
        "company": company,
        "count": count,
        "department": department,
        "role_keywords": role_keywords[:12],
        "seniority": seniority,
        "location": None,
        "intent": "find_people",
    }


async def gemini_json(prompt: str) -> Optional[dict]:
    if not GEMINI_CLIENT:
        return None

    def call():
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        return json.loads(text)

    try:
        result = await asyncio.to_thread(call)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


async def parse_intent(msg: str) -> dict:
    fallback = local_parse_intent(msg)

    prompt = f"""
Parse this lead-finding request into JSON.

User request:
{msg}

Return ONLY JSON with exactly these keys:
company: string
count: integer from 1 to 50
department: one of marketing, sales, engineering, product, finance, hr, operations, legal, or null
role_keywords: array of concise role/department keywords
seniority: one of c_level, vp, director, head, manager, lead, or null
location: string or null
intent: "find_people"

Rules:
- Do not invent a company.
- If the request says "10 people at Stripe in marketing", company is Stripe,
  count is 10, department is marketing.
- If no count is stated, use 10.
- If no department is stated, use null.
"""
    parsed = await gemini_json(prompt)

    if not parsed:
        return fallback

    # Sanitize Gemini output.
    parsed["company"] = normalize_space(str(parsed.get("company") or fallback["company"]))
    try:
        parsed["count"] = max(1, min(int(parsed.get("count", fallback["count"])), 50))
    except Exception:
        parsed["count"] = fallback["count"]

    dep = parsed.get("department")
    parsed["department"] = dep if dep in DEPARTMENT_ALIASES else fallback["department"]
    parsed["role_keywords"] = (
        parsed.get("role_keywords")
        if isinstance(parsed.get("role_keywords"), list)
        else fallback["role_keywords"]
    )
    parsed["seniority"] = (
        parsed.get("seniority")
        if parsed.get("seniority") in SENIORITY_TERMS
        else fallback["seniority"]
    )
    parsed["location"] = parsed.get("location") or fallback["location"]
    parsed["intent"] = "find_people"
    return parsed


# ----------------------------- Matching / ranking -----------------------------

def role_matches(role: str, intent: dict) -> bool:
    role_low = role.lower()

    dep = intent.get("department")
    if dep:
        aliases = DEPARTMENT_ALIASES.get(dep, [])
        if not any(alias.lower() in role_low for alias in aliases):
            return False

    seniority = intent.get("seniority")
    if seniority:
        terms = SENIORITY_TERMS.get(seniority, [])
        if not any(t in role_low for t in terms):
            return False

    return True


def deterministic_score(person: dict, intent: dict) -> int:
    role = person.get("role", "")
    score = 0

    if role_matches(role, intent):
        score += 100

    if person.get("email"):
        score += 40

    src = person.get("src", "")
    if src and same_domain(src, intent.get("_domain", "")):
        score += 15

    if re.search(r"\b(chief|vp|vice president|head|director)\b", role, re.I):
        score += 15

    if re.search(r"\b(founder|co-founder|ceo)\b", role, re.I):
        score += 10

    return score


async def llm_rank(people: list[dict], intent: dict) -> list[dict]:
    if not people or not GEMINI_CLIENT:
        return sorted(
            people,
            key=lambda p: deterministic_score(p, intent),
            reverse=True,
        )

    compact = [
        {
            "id": i,
            "name": p.get("name", ""),
            "role": p.get("role", ""),
            "email": p.get("email"),
            "source": p.get("src", ""),
        }
        for i, p in enumerate(people[:60])
    ]

    prompt = f"""
You are ranking already-discovered public professional contacts.

User intent:
{json.dumps(intent, ensure_ascii=False)}

Candidates:
{json.dumps(compact, ensure_ascii=False)}

Return ONLY a JSON object:
{{"ids":[integer,...]}}

Rank candidates by relevance to the requested department, role and seniority.
Prefer exact role matches and public exact emails.
NEVER create people or emails. Only return candidate IDs from the supplied list.
"""
    result = await gemini_json(prompt)

    if result and isinstance(result.get("ids"), list):
        by_id = {i: p for i, p in enumerate(people[:60])}
        ranked = [by_id[i] for i in result["ids"] if isinstance(i, int) and i in by_id]
        seen = {id(p) for p in ranked}
        ranked.extend(
            p for p in sorted(
                people,
                key=lambda x: deterministic_score(x, intent),
                reverse=True,
            )
            if id(p) not in seen
        )
        return ranked

    return sorted(
        people,
        key=lambda p: deterministic_score(p, intent),
        reverse=True,
    )


# ----------------------------- Email engine -----------------------------

def name_parts(name: str) -> tuple[str, str]:
    parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", name)
    if len(parts) < 2:
        return (parts[0] if parts else "", "")
    return parts[0], parts[-1]


def generate_candidates(first: str, last: str, domain: str) -> list[str]:
    if not first or not last:
        return []
    out = []
    seen = set()
    for p in PATTERNS:
        local = p.format(
            first=first.lower(),
            last=last.lower(),
            f=first[0].lower(),
        )
        email = f"{local}@{domain}"
        if email not in seen:
            seen.add(email)
            out.append(email)
    return out


def exact_public_email_for_person(person: dict, domain: str) -> Optional[str]:
    email = person.get("email")
    if email and email.lower().endswith("@" + domain.lower()):
        return email.lower()
    return None


async def enrich_email(person: dict, domain: str, mx: list[str]) -> dict:
    exact = exact_public_email_for_person(person, domain)
    first, last = name_parts(person.get("name", ""))

    if exact:
        return {
            "email": exact,
            "email_status": "verified",
            "email_how": "exact public email found on a public source",
        }

    candidates = generate_candidates(first, last, domain)
    if candidates:
        return {
            "email": candidates[0],
            "email_status": "inferred",
            "email_how": "generated from a common company email pattern; not mailbox-verified",
        }

    return {
        "email": None,
        "email_status": "unknown",
        "email_how": "no reliable public email found",
    }


# ----------------------------- API -----------------------------

class ChatIn(BaseModel):
    message: str


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "gemini_configured": bool(GEMINI_CLIENT),
        "gemini_model": GEMINI_MODEL,
    }


@app.post("/api/chat")
async def chat(body: ChatIn):
    steps = ["Understanding your request…"]
    intent = await parse_intent(body.message)

    company = intent.get("company", "").strip()
    if not company:
        return {
            "steps": steps,
            "results": [],
            "reply": "Tell me the company and who you want to find.",
        }

    steps.append(
        f"Searching for {intent.get('count', 10)} people at {company}"
        + (f" in {intent['department']}" if intent.get("department") else "")
        + "…"
    )

    domain = await find_domain(company)
    if not domain:
        return {
            "steps": steps + ["Could not resolve the company domain."],
            "results": [],
            "reply": f"I couldn't resolve the official domain for “{company}”. Try adding the company's website.",
        }

    intent["_domain"] = domain
    steps.append(f"Company domain: {domain}")

    mx = await mx_hosts(domain)
    steps.append(
        "Email receiving (MX) records found."
        if mx
        else "No MX records were found."
    )

    steps.append("Scanning public company/team pages…")
    people = await scrape_people(domain)

    if len(people) < intent["count"]:
        steps.append("Expanding discovery with public web search…")
        extra = await public_search_people(company, domain, intent)

        existing = {(p["name"].lower(), p["role"].lower()) for p in people}
        for p in extra:
            key = (p["name"].lower(), p["role"].lower())
            if key not in existing:
                existing.add(key)
                people.append(p)

    if not people:
        return {
            "steps": steps + ["No named public contacts were found."],
            "results": [],
            "reply": (
                f"I found {domain}, but couldn't find enough named people on "
                "public pages. Try a broader department or a different company query."
            ),
        }

    # Prefer exact requested department/seniority, but don't return nothing
    # if public data is sparse.
    matched = [p for p in people if role_matches(p.get("role", ""), intent)]
    pool = matched if matched else people

    steps.append(f"Found {len(people)} public candidate profiles.")

    ranked = await llm_rank(pool, intent)
    target = ranked[: max(intent["count"], 1)]

    results = []
    for p in target:
        email_data = await enrich_email(p, domain, mx)
        results.append({
            "name": p.get("name", ""),
            "role": p.get("role", "") or "Role not stated",
            "email": email_data["email"],
            "email_status": email_data["email_status"],
            "email_how": email_data["email_how"],
            "source": p.get("src", ""),
            "source_type": p.get("source_type", "public_web"),
        })

    verified = sum(1 for r in results if r["email_status"] == "verified")
    inferred = sum(1 for r in results if r["email_status"] == "inferred")

    steps.append(
        f"Prepared {len(results)} contacts: {verified} exact public emails, "
        f"{inferred} inferred emails."
    )

    reply = (
        f"Found {len(results)} relevant public contacts for {company}. "
        f"{verified} have exact public emails; {inferred} emails are inferred "
        "from naming patterns and are not mailbox-verified."
    )

    return {
        "steps": steps,
        "intent": intent,
        "domain": domain,
        "mx": bool(mx),
        "results": results,
        "reply": reply,
    }


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
