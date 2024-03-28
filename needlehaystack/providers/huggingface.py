import os
import asyncio
from operator import itemgetter
from typing import Optional
import torch

import huggingface_hub
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .model import ModelProvider

SYSTEM_PROMPT = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct"""

CONTEXT_PROMPT = """
<context>
{context}
</context>
"""

QUESTION_PROMPT = """
<question>
{question}
</question>
"""

class HuggingFace(ModelProvider):
    """
    A wrapper class for interacting with OpenAI's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the OpenAI model to use for evaluations and interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_new_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the HuggingFace model provider with a specific model.

        Args:
            model_name (str): The path of the HuggingFace model to use. Defaults to 'mistralai/Mistral-7B-Instruct-v0.2'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """
        api_key = os.getenv('NIAH_MODEL_API_KEY')
        if (not api_key):
            raise ValueError("NIAH_MODEL_API_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key

        huggingface_hub.login(self.api_key)
        
        self.model = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            device=0,
            task="text-generation",
            pipeline_kwargs=model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    async def evaluate_model(self, prompt_template: str) -> str:
        """
        Evaluates a given prompt using the OpenAI model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        prompt = PromptTemplate.from_template(
            prompt_template
        )

        chain = LLMChain(llm=self.model, prompt=prompt)

        response = await chain.ainvoke(input={})
        return response
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        prompt_format = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(context=context) + "\n\n" + QUESTION_PROMPT.format(question=retrieval_question) + "\n\nDon't give information outside the document or repeat your findings. Just provide the most relevant sentence in the context."
            }]
        
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        
        if "system" not in self.tokenizer.chat_template or "raise_exception('System role not supported')" in self.tokenizer.chat_template:
            prompt_format.pop(0)
            prompt_format[0]["content"] = SYSTEM_PROMPT + "\n" + prompt_format[0]["content"]

        return self.tokenizer.apply_chat_template(prompt_format, tokenize=False, add_generation_prompt=True)
    
    def encode_text_to_tokens(self, text: str, no_bos: bool = False) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.
            no_bos (bool): If True, the tokenization result will not contain bos token at first.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        if no_bos:
            return self.tokenizer.encode(text)[1:]
        else:
            return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the HuggingFace model, and returns the model's response. This method leverages the LangChain 
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

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings. Just provide the most relevant sentence in the context without any additional explanation."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = HuggingFacePipeline(
            model_id=self.model_name,
            device="auto",
            task="text-generation",
            model_kwargs=self.model_kwargs
        )

        chat_model = ChatHuggingFace(llm=model)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | chat_model 
                )
        return chain