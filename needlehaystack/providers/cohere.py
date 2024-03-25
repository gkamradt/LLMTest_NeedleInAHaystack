import os
import pkg_resources

from operator import itemgetter
from typing import Optional

from cohere import Client, AsyncClient

from .model import ModelProvider

class Cohere(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens          = 50,
                                      temperature           = 0.3)

    def __init__(self,
                 model_name: str = "command-r",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        :param model_name: The name of the model. Default is 'command-r'.
        :param model_kwargs: Model configuration. Default is {max_tokens_to_sample: 300, temperature: 0}
        """

        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key

        self.client = AsyncClient(api_key=self.api_key)

    async def evaluate_model(self, prompt: str) -> str:
        response = await self.client.chat(message=prompt[-1]["message"], chat_history=prompt[:-1], model=self.model_name, **self.model_kwargs)
        return response.text

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return [{
                "role": "System",
                "message": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "User",
                "message": context
            },
            {
                "role": "User",
                "message": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        if not text: return []
        return Client().tokenize(text=text, model=self.model_name).tokens

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        # Assuming you have a different decoder for Anthropic
        return Client().detokenize(tokens=tokens[:context_length], model=self.model_name).text
