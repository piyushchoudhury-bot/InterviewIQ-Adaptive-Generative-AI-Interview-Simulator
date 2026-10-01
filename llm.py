"""
llm.py
------
Thin wrapper around the Google GenAI SDK (Gemini) used everywhere else in
the application. Centralizing all LLM calls here means:

1. The rest of the codebase never touches the SDK directly.
2. We can add Demo Mode / error handling / JSON parsing in exactly one place.
3. If the API key is missing, the quota is exceeded, or the network fails,
   the app degrades gracefully instead of crashing.

The model name is configurable via the GEMINI_MODEL environment variable so
we never hard-code an obsolete model.
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

_client = None
_client_init_error = None


def _get_client():
    """Lazily create the GenAI client so importing this module never fails
    even if the SDK or API key is missing."""
    global _client, _client_init_error
    if _client is not None or _client_init_error is not None:
        return _client
    if not GEMINI_API_KEY:
        _client_init_error = "GEMINI_API_KEY is not set."
        return None
    try:
        from google import genai

        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception as e:
        _client_init_error = f"Could not initialize Gemini client: {e}"
        return None


def is_available() -> bool:
    """True only if a client can be created. Does NOT guarantee the network
    call will succeed (quota/network errors are handled per-call)."""
    return _get_client() is not None


def get_status_message() -> str:
    if GEMINI_API_KEY == "":
        return "No GEMINI_API_KEY found in environment. Running in Demo Mode is recommended."
    if _client_init_error:
        return _client_init_error
    return f"Connected to Gemini model: {GEMINI_MODEL}"


def generate_text(prompt: str, system_instruction: str = None, temperature: float = 0.7):
    """
    Calls Gemini with a plain prompt and returns (text, error).
    error is None on success.
    """
    client = _get_client()
    if client is None:
        return None, _client_init_error or "Gemini client unavailable."

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text, None
    except Exception as e:
        return None, f"Gemini API call failed: {e}"


def _extract_json_block(text: str):
    """Attempts to pull a JSON object/array out of arbitrary LLM text output,
    stripping markdown fences and any leading/trailing prose."""
    if text is None:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fall back to locating the first {...} or [...] block
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


def generate_json(prompt: str, system_instruction: str = None, temperature: float = 0.4):
    """
    Calls Gemini and asks for a structured JSON response.
    Returns (parsed_dict_or_list, error). error is None on success.
    A malformed response never raises -- it returns (None, error_message)
    so callers can fall back to deterministic defaults.
    """
    client = _get_client()
    if client is None:
        return None, _client_init_error or "Gemini client unavailable."

    json_instruction = (
        (system_instruction or "")
        + "\n\nYou must respond with ONLY valid JSON. No markdown fences, "
        "no commentary, no explanation before or after the JSON."
    )

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=json_instruction,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None)
        parsed = _extract_json_block(text)
        if parsed is None:
            return None, "Could not parse a valid JSON object from Gemini's response."
        return parsed, None
    except Exception as e:
        return None, f"Gemini API call failed: {e}"
