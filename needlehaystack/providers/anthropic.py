import os
import pkg_resources

from operator import itemgetter
from typing import Optional

from anthropic import AsyncAnthropic
from anthropic import Anthropic as AnthropicModel
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

from .model import ModelProvider

class Anthropic(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens_to_sample  = 300,
                                      temperature           = 0)

    def __init__(self,
                 model_name: str = "claude-2.1",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        :param model_name: The name of the model. Default is 'claude'.
        :param model_kwargs: Model configuration. Default is {max_tokens_to_sample: 300, temperature: 0}
        """

        if "claude" not in model_name:
            raise ValueError("If the model provider is 'anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key

        self.model = AsyncAnthropic(api_key=self.api_key)
        self.tokenizer = AnthropicModel().get_tokenizer()

        resource_path = pkg_resources.resource_filename('needlehaystack', 'providers/Anthropic_prompt.txt')

        # Generate the prompt structure for the Anthropic model
        # Replace the following file with the appropriate prompt structure
        with open(resource_path, 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        response = await self.model.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self.model_kwargs)
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
