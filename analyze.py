import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from litellm import completion

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

# LiteLLM reads provider keys from environment variables; Groq uses GROQ_API_KEY.
API_KEY_ENV_VAR = "GROQ_API_KEY"
REQUIRED_FIELDS = {
    "destination": str,
    "price_range": str,
    "ideal_visit_times": list,
    "top_attractions": list,
}

load_dotenv()


def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise ValueError(f"Missing API key. Set {API_KEY_ENV_VAR} in your environment.")

    system_prompt = (
        "You are a travel planner. Return only valid JSON with the required keys: "
        "destination, price_range, ideal_visit_times, top_attractions."
    )
    user_prompt = (
        f"Create a concise travel itinerary for {destination}. "
        "ideal_visit_times should be a short list of seasons or months. "
        "top_attractions should be a short list of major sights."
    )

    response = completion(
        model=MODEL,
        api_key=api_key,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON.") from exc

    _validate_itinerary_schema(data)
    data["destination"] = data.get("destination") or destination
    return data


def _validate_itinerary_schema(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object.")

    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"Model response missing required keys: {', '.join(missing)}")

    for key, expected_type in REQUIRED_FIELDS.items():
        if not isinstance(data[key], expected_type):
            raise ValueError(f"Field '{key}' must be of type {expected_type.__name__}.")

    for key in ("ideal_visit_times", "top_attractions"):
        if not all(isinstance(item, str) for item in data[key]):
            raise ValueError(f"Field '{key}' must be a list of strings.")
