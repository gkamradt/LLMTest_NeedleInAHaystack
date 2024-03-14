import pkg_resources

from operator import itemgetter
from typing import Optional

from anthropic import Anthropic as AnthropicModel
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate

from .model import ModelProvider

class Bedrock(ModelProvider):
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        :param model_id: The model ID. Default is 'anthropic.claude-3-sonnet-20240229-v1:0'.
        :param model_kwargs: Model configuration. Default is {max_tokens: 300, temperature: 0}
        """

        if "anthropic" not in model_name:
            raise NotImplementedError

        self.model_name = model_name
        self.model_kwargs = model_kwargs

        self.tokenizer = AnthropicModel().get_tokenizer()

        resource_path = pkg_resources.resource_filename('needlehaystack', 'providers/Anthropic_prompt.txt')

        # Generate the prompt structure for the Anthropic model
        # Replace the following file with the appropriate prompt structure
        with open(resource_path, 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        raise NotImplementedError

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
        model = BedrockChat(
            model_id=self.model_name,
            model_kwargs=self.model_kwargs,
        )
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt
                | model
                )
        return chain
