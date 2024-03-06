import uuid
from langsmith.client import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult, run_evaluator
from typing import Union
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

class LangSmithEvaluator():

    def __init__(self, model_name: str = "gpt-4-0125-preview", api_key: str = None):
        self.eval_model_name = model_name

    def evaluate_chain(self, chain, context_length, depth_percent, model_name):

        eval_model_name = self.eval_model_name

        @run_evaluator
        def score_relevance(run: Run, example: Union[Example, None] = None):
            
            print("--RUN IN LANGSMITH--")
            student_answer = run.outputs["output"]
            reference = example.outputs["answer"]
        
            # Grade prompt 
            template = """You are a grader grading a student response relative to a reference. \n 
                    The reference is a list of elements. The grade is the number of correctly returned \n 
                    elements. For example, if the reference has 5 elements and the student returns 3, \n
                    then the grade is 3. 
                    Here is the student answer: \n --- --- --- \n {answer}
                    Here is the reference question: \n --- --- --- \n {reference}"""
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
            model = ChatOpenAI(temperature=0, model=model_name)
            
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

            return EvaluationResult(key="check_execution", score=score[0].score)

        # Config
        evaluation_config = RunEvalConfig(
            custom_evaluators = [score_relevance],
        )

        client = Client()
        run_id = uuid.uuid4().hex[:4]
        project_name = "multi-needle-eval"
        client.run_on_dataset(
            dataset_name="multi-needle-eval",
            llm_or_chain_factory=chain,
            project_metadata={"context_length": context_length, 
                            "depth_percent": depth_percent, 
                            "model_name": model_name},
            evaluation=evaluation_config,
            project_name=f"{context_length}-{depth_percent}--{model_name}--{project_name}--{run_id}",
        )

