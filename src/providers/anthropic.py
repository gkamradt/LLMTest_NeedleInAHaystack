import os
from operator import itemgetter
from typing import Optional

from anthropic import AsyncAnthropic
from anthropic import Anthropic as AnthropicModel
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

from .model import ModelProvider

class Anthropic(ModelProvider):
    def __init__(self, model_name: str = "claude-2.1", api_key: str = None):
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
        self.tokenizer = AnthropicModel().get_tokenizer()

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
        return self.tokenizer.encode(text).ids
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        # Assuming you have a different decoder for Anthropic
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the Anthropic model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

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

        template = """You are a helpful assistant that answers user questions based on the context provided. \n
        Keep your response short and direct. If asked for a list, give a clearly numbered list with little \n
        preamble. Answer the question based only on the following context:
        \n ------- \n {context} \n ------- \n
        Here is the user question: \n --- --- --- \n {question}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatAnthropic(temperature=0, model=self.model_name)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain
