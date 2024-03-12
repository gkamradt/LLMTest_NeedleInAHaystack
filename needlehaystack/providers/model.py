
from abc import ABC, abstractmethod
from typing import Optional

class ModelProvider(ABC):
    @abstractmethod
    async def evaluate_model(self, prompt: str) -> str: ...

    @abstractmethod
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]: ...

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str: ...