import os
import pkg_resources

from operator import itemgetter
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere

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

    async def evaluate_model(self, prompt: tuple[str, list[dict, str, str]]) -> str:
        message, chat_history = prompt
        response = await self.client.chat(message=message, chat_history=chat_history, model=self.model_name, **self.model_kwargs)
        return response.text

    def generate_prompt(self, context: str, retrieval_question: str) -> tuple[str, list[dict[str, str]]]:
        '''
        Prepares a chat-formatted prompt
        Args:
            context (str): The needle in a haystack context
            retrieval_question (str): The needle retrieval question

        Returns:
            tuple[str, list[dict[str, str]]]: prompt encoded as last message, and chat history

        '''
        return (
            f"{retrieval_question} Don't give information outside the document or repeat your findings", 
            [{
                "role": "System",
                "message": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "User",
                "message": context
            }]
        )
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        if not text: return []
        return Client().tokenize(text=text, model=self.model_name).tokens

    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        # Assuming you have a different decoder for Anthropic
        return Client().detokenize(tokens=tokens[:context_length], model=self.model_name).text

    def get_langchain_runnable(self, context: str):
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """


        template = """Human: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        <document_content>
        {context} 
        </document_content>
        Here is the user question:
        <question>
         {question}
        </question>
        Don't give information outside the document or repeat your findings.
        Assistant: Here is the most relevant information in the documents:"""
        
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        model = ChatCohere(cohere_api_key=api_key, temperature=0.3, model=self.model_name)
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain
