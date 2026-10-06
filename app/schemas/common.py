import re
from typing import Annotated, Any

from pydantic import BeforeValidator

# Deliberately simple: good enough to pull an address out of LLM output or a form field.
EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _none_to_empty(value: Any) -> Any:
    return "" if value is None else value


def normalize_email_address(text: str) -> str | None:
    """The first email address found in `text` (e.g. in "Ana <ana@x.mk>"), or None."""
    match = EMAIL_ADDRESS.search(text)
    return match.group(0) if match else None


# Legacy rows were written with nullable columns; render them as empty strings instead of failing.
DisplayStr = Annotated[str, BeforeValidator(_none_to_empty)]
