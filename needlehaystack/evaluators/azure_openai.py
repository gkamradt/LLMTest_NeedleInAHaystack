from .openai import OpenAIEvaluator
from ..utils import get_from_env_or_error

from langchain_openai import AzureChatOpenAI


class AzureOpenAIEvaluator(OpenAIEvaluator):

    def __init__(
        self,
        model_name: str = None,
        model_kwargs: dict = OpenAIEvaluator.DEFAULT_MODEL_KWARGS,
        true_answer: str = None,
        question_asked: str = None,
        azure_openai_endpoint: str = None,
        azure_openai_api_version: str = "2024-02-01",
    ):
        """
        :param model_name: The deployment name of the model in Azure OpenAI Studio.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        :param azure_openai_endpoint: The endpoint of the Azure OpenAI resource.
        """

        if not model_name:
            raise ValueError("model_name must be supplied with init.")
        if (not true_answer) or (not question_asked):
            raise ValueError(
                "true_answer and question_asked must be supplied with init."
            )

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        self.api_key = get_from_env_or_error(
            env_key="NIAH_EVALUATOR_API_KEY",
            error_message="{env_key} must be in env for using openai evaluator."
        )

        self.azure_openai_endpoint = azure_openai_endpoint or get_from_env_or_error(
            env_key="AZURE_OPENAI_ENDPOINT",
            error_message="azure_openai_endpoint must be supplied with init or {env_key} must be in your env."
        )

        self.azure_openai_api_version = azure_openai_api_version or get_from_env_or_error(
            env_key="OPENAI_API_VERSION",
            error_message="azure_openai_api_version must be supplied with init or {env_key} must be in your env."
        )

        self.evaluator = AzureChatOpenAI(
            azure_deployment=self.model_name,
            azure_endpoint=self.azure_openai_endpoint,
            openai_api_version=self.azure_openai_api_version,
            api_key=self.api_key,
            **self.model_kwargs
        )
