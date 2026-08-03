from anima_search.annotation.validation import extract_json_object
from anima_search.schemas import SearchQuery


class QueryParser:
    def __init__(self, text_generator: object | None = None, prompt: str = "") -> None:
        self.text_generator = text_generator
        self.prompt = prompt

    def parse(self, query: str, generator: object | None = None) -> SearchQuery:
        source = generator or self.text_generator
        if source is None:
            return SearchQuery(raw_text=query)
        try:
            source = source() if callable(source) else source
            payload = extract_json_object(source.generate_text(f"{self.prompt}\n用户查询：{query}"))
            payload["raw_text"] = query
            return SearchQuery.model_validate(payload)
        except Exception:
            return SearchQuery(raw_text=query)
