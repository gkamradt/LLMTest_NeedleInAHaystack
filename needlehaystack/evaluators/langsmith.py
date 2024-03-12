from typing import Union
import uuid

from langchain_openai import ChatOpenAI  
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith.client import Client
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run

@run_evaluator
def score_relevance(run: Run, example: Union[Example, None] = None):
    """
    A custom evaluator function that grades the language model's response based on its relevance
    to a reference answer.

    Args:
        run (Run): The execution run containing the model's response.
        example (Union[Example, None]): An optional example containing the reference answer.

    Returns:
        EvaluationResult: The result of the evaluation, containing the relevance score.
    """
    
    print("--LANGSMITH EVAL--")
    #print("--MODEL: ", model_name)
    #print("--EVAL SET: ", eval_set)
    student_answer = run.outputs["output"]
    reference = example.outputs["answer"]

    # Grade prompt 
    template = """You are an expert grader of student answers relative to a reference answer. \n 
            The reference answer is a single ingredient or a list of ingredients related to pizza \n 
            toppings. The grade is the number of correctly returned ingredient relative to the reference. \n 
            For example, if the reference has 5 ingredients and the student returns 3, then the grade is 3. \n
            Here is the student answer: \n --- --- --- \n {answer}
            Here is the reference answer: \n --- --- --- \n {reference}"""
    # Prompt 
    prompt = PromptTemplate(
            template=template,
            input_variables=["answer", "reference"],
        )

    # Data model
    class grade(BaseModel):
        """Grade output"""
        score: int = Field(description="Score from grader")
    
    ## LLM
    # Use most performant model as grader
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    
    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)
    
    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )
    
    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])
    
    chain = (
            prompt
            | llm_with_tool 
            | parser_tool
        )

    score = chain.invoke({"answer":student_answer,
                            "reference":reference})

    return EvaluationResult(key="needles_retrieved", score=score[0].score)

class LangSmithEvaluator():
    """
    An evaluator class that leverages the LangSmith API for evaluating language models' performance on specific tasks. 
    This class primarily focuses on evaluating the ability of a language model to retrieve and accurately present information 
    from a provided context (the "needle" in a "haystack").
    """

    def evaluate_chain(self, chain, context_length, depth_percent, model_name, eval_set, num_needles, needles, insertion_percentages):
        """
        Evaluates a language model's chain of operations, specifically focusing on the model's ability to 
        retrieve information accurately from a given context. This method defines a custom evaluator that
        grades the language model's responses based on relevance to a reference answer.

        Args:
            chain: The LangChain runnable or chain of operations to be evaluated.
            context_length (int): The length of the context in tokens.
            depth_percent (float): The percentage depth in the context where the information (needle) is located.
            model_name (str): The name of the language model being evaluated.
            eval_set (str): The evaluation set identifier, used to categorize and reference the evaluation.
            num_needles (int): The number of needles in the haystack. 
            needles (list[str]): The needles inserted into the haystack. 
            insertion_percentages (list[float]): The location of each needle in the haystack. 


        Details:
            The evaluation involves creating a grading prompt that asks the model to grade student responses
            based on their relevance to a given reference answer. This approach allows for quantifying the
            model's accuracy in retrieving and synthesizing information from the provided context.
        """
        # Config
        evaluation_config = RunEvalConfig(
            custom_evaluators = [score_relevance],
        )

        client = Client()
        run_id = uuid.uuid4().hex[:4]
        project_name = eval_set
        client.run_on_dataset(
            dataset_name=eval_set,
            llm_or_chain_factory=chain,
            project_metadata={"context_length": context_length, 
                            "depth_percent": depth_percent, 
                            "num_needles": num_needles,
                            "needles": needles,
                            "insertion_percentages": insertion_percentages,
                            "model_name": model_name},
            evaluation=evaluation_config,
            project_name=f"{context_length}-{depth_percent}--{model_name}--{project_name}--{run_id}",
        )

