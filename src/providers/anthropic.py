import os

from .model import ModelProvider

from anthropic import AsyncAnthropic, Anthropic
from typing import Optional

class Anthropic(ModelProvider):
    def __init__(self, model_name: str = "claude", api_key: str = None):
        """
        :param model_name: The name of the model. Default is 'claude'.
        :param api_key: The API key for Anthropic. Default is None.
        """

        if "claude" not in model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        if (api_key is None) and (not os.getenv('ANTHROPIC_API_KEY')):
            raise ValueError("Either api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")

        self.model_name = model_name
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        self.model = AsyncAnthropic(api_key=self.api_key)
        self.enc = Anthropic().get_tokenizer()

        # Generate the prompt structure for the Anthropic model
        # Replace the following file with the appropriate prompt structure
        with open('Anthropic_prompt.txt', 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        response = await self.model.completions.create(
            model=self.model_name,
            max_tokens_to_sample=300,
            prompt=prompt,
            temperature=0)
        return response.completion

    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return self.prompt_structure.format(
            retrieval_question=retrieval_question,
            context=context)
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.enc.encode(text).ids
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        # Assuming you have a different decoder for Anthropic
        return self.enc.decode(tokens[:context_length])