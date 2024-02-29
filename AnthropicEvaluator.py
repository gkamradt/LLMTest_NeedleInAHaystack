import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from anthropic import AsyncAnthropic, Anthropic


class AnthropicEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        if 'anthropic_api_key' not in  kwargs and not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env")

        if 'model_name' not in kwargs:
            raise ValueError("model_name must be supplied with init")
        elif "claude" not in kwargs['model_name']:
            raise ValueError(
                "If the model provider is 'Anthropic', the model name must include 'claude'. "
                "See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")

        if 'evaluation_method' not in kwargs:
            print("since evaluation method is not specified , default method substring_match will be used for evaluation")
        elif kwargs['evaluation_method'] not in ('gpt4', 'substring_match'):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")
        elif kwargs['evaluation_method'] == 'gpt4' and 'openai_api_key' not in  kwargs and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("if evaluation_method is gpt4 , openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env")
        else:
            self.openai_api_key= kwargs.get('openai_api_key', os.getenv('OPENAI_API_KEY'))

        self.anthropic_api_key = kwargs.pop('anthropic_api_key', os.getenv('ANTHROPIC_API_KEY'))
        self.model_name = kwargs['model_name']
        self.model_to_test_description = kwargs.pop('model_name')
        self.model_to_test = AsyncAnthropic(api_key=self.anthropic_api_key)
        self.tokenizer = Anthropic().get_tokenizer()


        super().__init__(**kwargs)

    def get_encoding(self,context):
        return self.tokenizer.encode(context).ids

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context):
        return [
            {
                "role": "user",
                "content": f"{context}\n\n {self.retrieval_question} Don't give information outside the document or repeat your findings"
            },
            {
                "role": "assistant",
                "content": "Here is the most relevant sentence in the context:"
            }
        ]


    async def get_response_from_model(self, prompt):
        response = await self.model_to_test.messages.create(
            model=self.model_name,
            messages =prompt,
            system="You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
            max_tokens=300,
            temperature=0
        )
        return response.content[0].text



if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = AnthropicEvaluator(model_name='claude-2.1', evaluation_method='substring_match')

    ht.start_test()





