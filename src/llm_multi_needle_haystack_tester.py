import asyncio
import glob
import json
import numpy as np
import os
import time

from .evaluators import Evaluator
from .providers import ModelProvider

from .llm_needle_haystack_tester import LLMNeedleHaystackTester

from asyncio import Semaphore
from datetime import datetime, timezone

class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    def __init__(self, *args, 
                 needles=[], 
                 model_to_test: ModelProvider = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status = True,
                 eval_set = "multi-needle-eval-sf",
                 **kwargs):
        """
        Initialize the LLMMultiNeedleHaystackTester with the capability to insert multiple needles.
        
        :param needles: A list of needles (random facts) to insert into the context.
        """
        super().__init__(*args, model_to_test=model_to_test, **kwargs)
        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles into the context at specified depth percentages.
        
        :param context: The original context string.
        :param depth_percent: The depth percent at which to insert the needles.
        :param context_length: The total length of the context in tokens.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]

        # Insert needles at calculated points
        for needle in self.needles:
            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)
            # Insert each needle at its corresponding depth percentage
            # For simplicity, evenly distribute needles throughout the context
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
            depth_percent += (100 - depth_percent) / len(self.needles)  # Adjust depth for next needle

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def encode_and_trim(self, context, context_length):
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context

    async def generate_context(self, context_length, depth_percent):
        """
        Overrides the generate_context method to insert multiple needles.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = await self.insert_needles(context, depth_percent, context_length)
        return context
    
    async def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        test_start_time = time.time()

        # LangSmith
        if self.evaluator.__class__.__name__ == "LangSmithEvaluator":
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name, self.eval_set)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        else:
            print("EVALUATOR: OpenAI Model")
            # Prepare your message to send to the model you're going to evaluate
            prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            # Go see if the model can answer the question to pull out your random fact
            response = await self.model_to_test.evaluate_model(prompt)
            # Compare the reponse to the actual needle you placed
            score = self.evaluation_model.evaluate_response(response)

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

            results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
            }

            self.testing_results.append(results)

            if self.print_ongoing_status:
                print (f"-- Test Summary -- ")
                print (f"Duration: {test_elapsed_time:.1f} seconds")
                print (f"Context: {context_length} tokens")
                print (f"Depth: {depth_percent}%")
                print (f"Score: {score}")
                print (f"Response: {response}\n")

            context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

            if self.save_contexts:
                results['file_name'] = context_file_location

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

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())