import re
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

import pyjson5
from pydantic import BaseModel

from llm_tools.utils.log import LogFactory


_log = LogFactory.get_logger(__name__)


T = TypeVar("T", bound=BaseModel)


# TODO: temporary fix, since some models escape JSON strings inside arrays.
def _fix_escaped_strings(text: str) -> str:
    text = re.sub(r'\\+"', '"', text)

    return text


class StructuredOutputParser:
    _fixing_functions: list[Callable[[str], str]] = [
        _fix_escaped_strings,
    ]
    _schema_keys: set[str] = {"properties", "title", "type"}

    @overload
    def parse(
        self,
        text: str | None,
    ) -> dict: ...

    @overload
    def parse(
        self,
        text: str | None,
        model: type[T],
        lenient: bool = True,
    ) -> T: ...

    @overload
    def parse(
        self,
        text: str | None,
        model: None = None,
        lenient: bool = True,
    ) -> dict: ...

    def parse(
        self,
        text: str | None,
        model: type[T] | None = None,
        lenient: bool = True,
    ) -> dict | T:
        parsed = self._parse_json(text, try_fix=lenient)

        if model is not None:
            try:
                parsed = model(**parsed)
            except Exception as e:
                if lenient:
                    parsed = self._extract_properties(parsed)
                    parsed = model(**parsed)
                else:
                    print(f"Could not parse model [input='{parsed}']")
                    raise e

        return parsed

    def _parse_json(self, text: str | None, try_fix: bool) -> dict[str, Any]:
        if text is None or not text.strip():
            return {}

        parsed = None
        try:
            parsed = self._parse_raw_json(text)
        except Exception as e:
            if try_fix:
                for idx, func in enumerate(self._fixing_functions, start=1):
                    _log.warning(
                        "Could not parse JSON. Trying to fix. "
                        "[iteration='{}', text='{}', error='{}']",
                        idx,
                        text,
                        e,
                    )
                    print(
                        "Could not parse JSON. Trying to fix. "
                        f"[iteration='{idx}', text='{text}', error='{e}']"
                    )
                    try:
                        text = func(text)
                        parsed = self._parse_raw_json(text)
                    except Exception:  # noqa
                        pass

        if parsed is None:
            parsed = {}
            _log.error("Could not parse JSON. [text='{}']", text)

        return parsed

    @staticmethod
    def _parse_raw_json(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]

        return pyjson5.decode(text)

    @classmethod
    def _extract_properties(cls, parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Extract properties from dict

        Since LLMs sometimes respond with a schema object rather than than the
        actual properties, this method tries to extract them.
        """

        if parsed.keys() & cls._schema_keys:
            parsed = cast(dict, parsed["properties"])

        return parsed
