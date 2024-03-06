from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from jsonargparse import CLI

from src import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from src.evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from src.providers import Anthropic, ModelProvider, OpenAI

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
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name, api_key=args.api_key)
        case "anthropic":
            return Anthropic(model_name=args.model_name, api_key=args.api_key)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(question_asked=args.retrieval_question,
                                   true_answer=args.needle,
                                   api_key=args.evaluator_api_key)
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def get_model_to_test(args: CommandArgs) -> ModelProvider:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name, api_key=args.api_key)
        case "anthropic":
            return Anthropic(model_name=args.model_name, api_key=args.api_key)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
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
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.

    Example usage:
    python main.py --evaluator langsmith --context_lengths_num_intervals 3 --document_depth_percent_intervals 3 
    --provider openai --model_name "gpt-4-0125-preview" --multi_needle True --eval_set multi-needle-eval-pizza
    --needles '["Figs are one of the three most delicious pizza toppings.", 
    "Prosciutto is one of the three most delicious pizza toppings.", 
    "Goat cheese is one of the three most delicious pizza toppings."]'
    """
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