import os
from operator import itemgetter
from typing import Optional

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer 

from .openai import OpenAI


class Databricks(OpenAI):
    """
    A wrapper class for interacting with databrick's foundation model API via the existing wrapper OpenAI class

    Attributes:
        base_url (str): workspace URL where model endpoint is spun-up
        model_name (str): The name of the databricks foundation model to use for interactions.
        model (AsyncOpenAI): An instance of the AsyncOpenAI client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """

    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0.01)

    def __init__(self,
                 base_url: str,
                 model_name: str,
                 model_kwargs:str = DEFAULT_MODEL_KWARGS):
        """
        Initializes the databricks foundation model provider with a specific model.

        Args:
            base_url (str): workspace URL where model endpoint is spun-up.
            model_name (str): The name of the databricks foundation model to use for interactions.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.

        Raises:
            ValueError: If NIAH_MODEL_API_KEY is not found in the environment.
        """

        # Initiate parent class
        super().__init__(model_name, model_kwargs)
        self._base_url = base_url
        
        # Override model
        self.model = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url)
        
        # Set tokenizer
        if model_name == "databricks-dbrx-instruct":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", token=os.getenv('HF_PAT_KEY'))
        else:
            # Guesstimate - CHANGE THIS IN CASE WRONG TOKENIZER IS GETTING PULLED
            model_tags = model_name.split("-")
            provider = model_tags[0]
            model = "-".join(model_tags[1::])
            self.tokenizer = AutoTokenizer.from_pretrained(f"{provider}/{model}", trust_remote_code=True)
        
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "user",
                "content": f"<context> {context} </context> {retrieval_question} Don't give information outside the context or repeat your findings"
            }]
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the OpenAI model, and returns the model's response. This method leverages the LangChain 
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
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatOpenAI(
            temperature=0,
            model=self.model_name,
            base_url=self._base_url,
            api_key=self.api_key
            )
        
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain