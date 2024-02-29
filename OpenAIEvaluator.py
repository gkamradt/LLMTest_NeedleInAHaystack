import os
import tiktoken
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from openai import AsyncOpenAI


class OpenAIEvaluator(LLMNeedleHaystackTester):
    def __init__(self,**kwargs):
        if 'openai_api_key' not in  kwargs and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env")

        if 'model_name' not in kwargs:
            raise ValueError("model_name must be supplied with init, accepted model_names are 'gpt-4-1106-preview'")
        elif kwargs['model_name'] not in ['gpt-4-1106-preview']:
            raise ValueError("Model name must be in this list (gpt-4-1106-preview)")

        if 'evaluation_method' not in kwargs:
            print("since evaluation method is not specified , 'gpt4' will be used for evaluation")
        elif kwargs['evaluation_method'] not in ('gpt4', 'substring_match'):
            raise ValueError("evaluation_method must be 'substring_match' or 'gpt4'")


        self.openai_api_key = kwargs.pop('openai_api_key', os.getenv('OPENAI_API_KEY'))
        self.model_name = kwargs['model_name']
        self.model_to_test_description = kwargs.pop('model_name')
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.model_to_test = AsyncOpenAI(api_key=self.openai_api_key)

        super().__init__(**kwargs)



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

    async def get_response_from_model(self, prompt):
        response = await self.model_to_test.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=300,
            temperature=0
        )
        return response.choices[0].message.content




if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = OpenAIEvaluator(model_name='gpt-4-1106-preview', evaluation_method='gpt4')
    ht.start_test()
