import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from openai import AsyncOpenAI


class OpenAIEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        kwargs['model_provider'] = "OpenAI"
        if 'openai_api_key' not in  kwargs and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env")
        else:
            self.openai_api_key = kwargs.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
            self.model_name = kwargs['model_name']
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            self.model_to_test = AsyncOpenAI(api_key=self.openai_api_key)
            self.model_to_test_description = kwargs['model_name']


        if 'model_name' not in  kwargs:
            raise ValueError("model_name must be supplied with init")

        if 'evaluation_method' not in kwargs:
            print("since evaluation method is not specified , default method substring_match will be used for evaluation")
        elif kwargs['evaluation_method'] not in ('gpt4', 'substring_match'):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")

        super().__init__()

    def get_encoding(self,context):
        return self.tokenizer.encode(context)

    def get_decoding(self, encoded_context):
        return self.tokenizer.decode(encoded_context)

    def get_prompt(self, context):
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings"
            }
        ]





if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = OpenAIEvaluator(model_name='gpt-4-1106-preview', evaluation_method='gpt4')

    ht.start_test()





