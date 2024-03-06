import os
import tiktoken

from .model import ModelProvider

from operator import itemgetter
from openai import AsyncOpenAI
from typing import Optional
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate

class OpenAI(ModelProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", api_key: str = None):
        """
        :param model_name: The name of the model. Default is 'gpt-3.5-turbo-0125'.
        :param api_key: The API key for OpenAI. Default is None.
        """
        
        if (api_key is None) and (not os.getenv('OPENAI_API_KEY')):
            raise ValueError("Either api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")

        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        self.model = AsyncOpenAI(api_key=self.api_key)
        print(f"Using OpenAI model: {self.model_name}")
        self.enc = tiktoken.encoding_for_model(self.model_name)
    
    async def evaluate_model(self, prompt: str) -> str:
        response = await self.model.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
        return response.choices[0].message.content
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": context
            },
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.enc.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        return self.enc.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:


        template = """You are a helpful AI bot that answers questions for a user given the context provided.\n 
        Keep your response short and direct. Don't give information outside the document or repeat your findings. Here is your context: 
        \n ------- \n {context} \n ------- \n
        Here is the user question: \n --- --- --- \n {question}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatOpenAI(temperature=0, model=self.model_name)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain