from src import LLMNeedleHaystackTester
from src.providers import ModelProvider, Anthropic, OpenAI
from src.evaluators import Evaluator, OpenAIEvaluator

from dataclasses import dataclass
from dotenv import load_dotenv
from jsonargparse import CLI
from typing import Optional

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
    context_lengths_max: Optional[int] = 200000
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

def get_model_to_test(args: CommandArgs) -> ModelProvider:
    match args.provider.lower():
        case "openai":
            return OpenAI(api_key=args.api_key)
        case "anthropic":
            return Anthropic(api_key=args.api_key)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(question_asked=args.retrieval_question,
                                   true_answer=args.needle,
                                   api_key=args.evaluator_api_key)
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    args = CLI(CommandArgs, as_positional=False)

    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)

    tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()