from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()

def read_files(directory):
    context = ""
    for file in glob.glob(directory):
        with open(file, 'r') as f:
            context += f.read()
    return context

def encode_and_trim(context, context_length, enc):
    tokens = enc.encode(context)
    if len(tokens) > context_length:
        context = enc.decode(tokens[:context_length])
    return context

def insert_needle(needle, context, depth_percent, context_length, enc):
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 150

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
        period_tokens = enc.encode('.')
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = enc.decode(tokens_new_context)
    return new_context

def generate_context(needle, context_length, depth_percent):
    # Load up tiktoken so we navigate tokens more easily
    enc = tiktoken.encoding_for_model("gpt-4-1106-preview")

    # Get your Paul Graham files loaded into a string
    context = read_files("paulgrahamessays/*.txt")

    # Truncate the Paul Graham essays to the context length you desire
    context = encode_and_trim(context, context_length, enc)

    # Insert your random statement according to your depth percent
    context = insert_needle(needle, context, depth_percent, context_length, enc)

    return context

def evaluate_response(response, needle, question_to_ask, evaluation_model):
    accuracy_criteria = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Keep your explanations extremely short, just give the score
        """
    }

    # Using GPT-4 to evaluate
    evaluator = load_evaluator(
        "labeled_score_string",
        criteria=accuracy_criteria,
        llm=evaluation_model,
    )

    eval_result = evaluator.evaluate_strings(
        # The models response
        prediction=response,

        # The actual answer
        reference=needle,

        # The question asked
        input=question_to_ask,
    )

    return int(eval_result['score'])

def result_exists(results, context_length, depth_percent, version, model):
    """
    Checks to see if a result has already been evaluated or not
    """
    conditions_met = []
    for result in results:
        context_length_met = result['context_length'] == context_length
        depth_percent_met = result['depth_percent'] == depth_percent
        version_met = result.get('version', 1) == version
        model_met = result['model'] == model
        conditions_met.append(context_length_met and depth_percent_met and version_met)
    return any(conditions_met)

if __name__ == "__main__":
    needle = """
    The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.
    """
    question_to_ask = "What is the most fun thing to do in San Francisco?"

    # The code will check to see if a context_length, depth percent and version number have already been checked yet
    # Change the version # if you would like to run the results multiple times.
    # If you're just testing, then leave as version=1
    results_version = 1 

    # This will produce a list of context lengths for each experiment iteration. Make sure the max context length is within the bounds of your models limits.
    context_lengths = np.round(np.linspace(1000, 128000, num=15, endpoint=True)).astype(int)

    # This will product a list of document depths to place your random statement (needle) at.
    # Suggestion: Try out different distributions (like a sigmoid) to test non-evenly space intervals
    document_depth_percents = np.round(np.linspace(0, 100, num=15, endpoint=True)).astype(int)

    # The model we are testing. As of now it's set up for chat models with OpenAI
    # model_to_test = ChatOpenAI(model='gpt-4-1106-preview', temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey'))
    model_to_test = ChatAnthropic(model='claude-2', temperature=0, anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', 'YourAPIKey'))

    # This will get logged on your results
    model_to_test_description = 'claude-2'

    evaluation_model  = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY', 'YourAPIKey'))

    # Run through each iteration of context_lengths and depths
    for context_length in context_lengths:
        for depth_percent in document_depth_percents:
            # Load results from file. 
            try:
                with open('results.json', 'r') as f:
                    results = json.load(f)
            except FileNotFoundError:
                results = []
                pass

            # Checks to see if you've already checked a length/percent/version.
            # This helps if the program stop running and you want to restart later
            if result_exists(results, context_length, depth_percent, results_version, model_to_test_description):
                continue

            # Go generate the required length context and place your needle statement in
            context = generate_context(needle, context_length, depth_percent)

            # Prepare your message to send to the model you're going to evaluate
            messages = [
                SystemMessage(
                    content="You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                ),
                HumanMessage(
                    # This is the PG essays with your needle/random statement placed in it
                    # This is your haystack with a needle placed in it.
                    content=context
                ),
                HumanMessage(
                    # This is the question you'll ask to the model to tr≠≠y and retrieve your random statement/needle.
                    content="What is the most fun thing to do in San Francico based on the context? Don't give information outside the document or repeat your findings"
                ),
            ]

            # Go see if the model can answer the question to pull out your random fact
            response = model_to_test(messages)

            # Compare the reponse to the actual needle you placed
            score = evaluate_response(response, needle, question_to_ask, evaluation_model)

            results.append({
                # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
                'model' : model_to_test_description,
                'context_length' : int(context_length),
                'depth_percent' : int(depth_percent),
                'version' : results_version,
                'needle' : needle,
                'model_response' : response.content,
                'score' : score
            })

            print (f"Result #: {len(results)}/{len(context_lengths) * len(document_depth_percents)}")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response.content}\n")

            # Save results to a JSON file each run
            with open('results.json', 'w') as f:
                json.dump(results, f)

            # Optional. Sleep for a bit to stay under the rate limit
            # Rate limit is 150K tokens/min so it's set at 120K for some cushion
            sleep_time = (context_length / 120000)*60
            # print (f"Sleeping: {sleep_time}\n")
            time.sleep(sleep_time)