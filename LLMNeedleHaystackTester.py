from dotenv import load_dotenv
import os
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI

import numpy as np

import asyncio
from asyncio import Semaphore
from datetime import datetime, timezone
import time

from abc import ABC, abstractmethod

load_dotenv()


class LLMNeedleHaystackTester(ABC):
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 substr_validation_words=['dolores', 'sandwich'],
                 results_version=1,
                 context_lengths_min=1000,
                 context_lengths_max=200000,
                 context_lengths_num_intervals=35,
                 context_lengths=None,
                 document_depth_percent_min=0,
                 document_depth_percent_max=100,
                 document_depth_percent_intervals=35,
                 document_depth_percents=None,
                 document_depth_percent_interval_type="linear",
                 num_concurrent_requests=1,
                 save_results=True,
                 save_contexts=True,
                 final_context_length_buffer=200,
                 seconds_to_sleep_between_completions=None,
                 print_ongoing_status=True,
                 evaluation_method='gpt4'):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param substr_validation_words: If you choose substring evaluation of LLM response, presence of these list of keywords are verified to determine if the LLM respone is correct or not
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        :param evaluation_method: Choose between gpt to evaluate (get the score 1,3,5,7,10) else using simple substring matching , default is gpt4
        """
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.substr_validation_words = substr_validation_words
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.evaluation_method = evaluation_method

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(
                    np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals,
                                endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(
                        np.linspace(document_depth_percent_min, document_depth_percent_max,
                                    num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in
                                                    np.linspace(document_depth_percent_min, document_depth_percent_max,
                                                                document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        self.evaluation_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=self.openai_api_key)

        if evaluation_method == 'substring_match' and not all(
                word.lower() in needle.lower() for word in substr_validation_words):
            raise ValueError("You choose substring evaluation method but some of the words in substr_validation_words is not in the needle you provided"
                             f"\n\nneedle: {needle}"
                             f"\nsubstr_validation_words: {substr_validation_words}")

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    @abstractmethod
    def get_prompt(self, context):
        pass

    @abstractmethod
    async def get_response_from_model(self, prompt):
        pass

    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.get_prompt(context)

        test_start_time = time.time()

        # Go see if the model can answer the question to pull out your random fact
        response = await self.get_response_from_model(prompt)


        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # Compare the reponse to the actual needle you placed
        score = self.evaluate_response(response)

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model': self.model_to_test_description,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'

        if self.save_contexts:
            results['file_name']: context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists('results'):
                os.makedirs('results')

            # Save the result to file for retesting
            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

        if self.seconds_to_sleep_between_completions:
            await asyncio.sleep(self.seconds_to_sleep_between_completions)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    async def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.get_encoding(self.needle)
        tokens_context = self.get_encoding(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.get_encoding('.')

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.get_decoding(tokens_new_context)
        return new_context

    def evaluate_response(self, response):
        if self.evaluation_method == 'gpt4':
            return self.evaluate_response_gpt4(response)
        else:
            return self.evaluate_response_substring_match(response)

    def evaluate_response_gpt4(self, response):
        accuracy_criteria = {
            "accuracy": """
            Score 1: The answer is completely unrelated to the reference.
            Score 3: The answer has minor relevance but does not align with the reference.
            Score 5: The answer has moderate relevance but contains inaccuracies.
            Score 7: The answer aligns with the reference but has minor omissions.
            Score 10: The answer is completely accurate and aligns perfectly with the reference.
            Only respond with a numberical score
            """
        }

        # Using GPT-4 to evaluate
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=accuracy_criteria,
            llm=self.evaluation_model,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.needle,

            # The question asked
            input=self.retrieval_question,
        )

        return int(eval_result['score'])

    def evaluate_response_substring_match(self, response):
        response_lower = response.lower()
        if all(word in response_lower for word in self.substr_validation_words):
            return 1
        else:
            return 0

    @abstractmethod
    def get_encoding(self, context):
        pass

    @abstractmethod
    def get_decoding(self, encoded_context):
        pass

    def get_context_length_in_tokens(self, context):
        return len(self.get_encoding(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        encoded_context = self.get_encoding(context)
        return self.get_decoding(encoded_context[:context_length])

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = LLMNeedleHaystackTester()

    ht.start_test()