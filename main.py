from src import LLMNeedleHaystackTester
from src import LLMMultiNeedleHaystackTester 

from src.providers import ModelProvider, Anthropic, OpenAI
from src.evaluators import Evaluator, OpenAIEvaluator, LangSmithEvaluator

from dataclasses import dataclass, field
from dotenv import load_dotenv
from jsonargparse import CLI
from typing import Optional, List

load_dotenv()

@dataclass
class CommandArgs():
    provider: str = "openai"
    evaluator: str = "openai"
    api_key: Optional[str] = None
    evaluator_api_key: Optional[str] = None
    needle: Optional[str] = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 16000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[list[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 35
    document_depth_percents: Optional[list[int]] = None
    document_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    model_name: str = "gpt-3.5-turbo-0125" 
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-sf"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: List[str] = field(default_factory=lambda: [
        "The top ranked thing to do in San Francisco is go to Dolores Park.",
        "The second rated thing to do in San Francisco is eat at Tony's Pizza Napoletana.",
        "The third best thing to do in San Francisco is visit Alcatraz.",
        "The fourth recommended thing to do in San Francisco is hike up Twin Peaks.",
        "The fifth top activity in San Francisco is bike across the Golden Gate Bridge."
    ])

def get_model_to_test(args: CommandArgs) -> ModelProvider:
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name, api_key=args.api_key)
        case "anthropic":
            return Anthropic(model_name=args.model_name, api_key=args.api_key)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(question_asked=args.retrieval_question,
                                   true_answer=args.needle,
                                   api_key=args.evaluator_api_key)
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()