from typing import Annotated, Any

from pydantic import BeforeValidator


def _none_to_empty(value: Any) -> Any:
    return "" if value is None else value


# Legacy rows were written with nullable columns; render them as empty strings instead of failing.
DisplayStr = Annotated[str, BeforeValidator(_none_to_empty)]
