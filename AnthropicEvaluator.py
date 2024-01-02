import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from anthropic import AsyncAnthropic, Anthropic


class AnthropicEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        kwargs['model_provider'] = "Anthropic"
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

        self.anthropic_api_key = kwargs.get('anthropic_api_key', os.getenv('ANTHROPIC_API_KEY'))
        self.model_name = kwargs['model_name']
        self.model_to_test = AsyncAnthropic(api_key=self.anthropic_api_key)
        self.tokenizer = Anthropic().get_tokenizer()
        self.model_to_test_description = kwargs['model_name']

        super().__init__()

    def get_encoding(self,context):
        return self.tokenizer.encode(context)

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context):
        with open('Anthropic_prompt.txt', 'r') as file:
            prompt = file.read()
        return prompt.format(retrieval_question=self.retrieval_question, context=context)



if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = AnthropicEvaluator(model_name='gpt-4-1106-preview', evaluation_method='gpt4')

    ht.start_test()





