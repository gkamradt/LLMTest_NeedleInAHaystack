import subprocess
import sys

from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from anthropic import AsyncAnthropic, Anthropic
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI
import pandas as pd
import random
import nest_asyncio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phoenix.experimental.evals.models import LiteLLMModel
from classify import llm_classify
import asyncio
from phoenix.experimental.evals import (
    OpenAIModel,
    download_benchmark_dataset,
    llm_classify,
)

from datetime import datetime, timezone
import time

load_dotenv()

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the special magic {} number?",
                 results_version = 1,
                 rnd_number_digits = 7,
                 context_lengths_min = 1000,
                 context_lengths_max = 126000,
                 #context_lengths_num_intervals = 5,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 #document_depth_percent_intervals = 5,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 #model_provider = "Anthropic",
                 openai_api_key=None,
                 #anthropic_api_key = None,
                 model_name='gpt-4-1106-preview',
                 #model_name='claude-2.1',
                 save_results = False,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param rnd_number_digits: The number of digits in the random number. Default is 7.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
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
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        self.rnd_number_digits = rnd_number_digits
        needle="\nThe special magic {city} number is: {rnd_number}\n"
        self.needle = needle
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        if model_provider not in ["OpenAI", "Anthropic", "Anyscale"]:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
        if model_provider == "Anthropic" and "claude" not in model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
            raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
        else:
            self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            
        if not self.model_name:
            raise ValueError("model_name must be provided.")
        
        if model_provider == "OpenAI":
            self.model_to_test = AsyncOpenAI(api_key=self.openai_api_key)
            self.enc = tiktoken.encoding_for_model(self.model_name)
        elif model_provider == "Anthropic":
            self.model_to_test = AsyncAnthropic(api_key=self.anthropic_api_key)
            self.enc = Anthropic().get_tokenizer()
        
        self.model_to_test_description = model_name
        self.evaluation_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = self.openai_api_key)

    def generate_random_number(self, num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    ANTHROPIC_TEMPLATE = '''Human: You are a close-reading bot with a great memory who answers questions for users. I'm going to give you the text of some essays. Amidst these essays ("the haystack") I've inserted a sentence ("the needle") that contains an answer to the user's question. Here's the question:
                <question>{question}</question>
                Here's the text of the essays. The answer appears in it somewhere.
                <haystack>
                {context}
                </haystack>
                Now that you've read the context, please answer the user's question, repeated one more time for ease of reference:
                <question>{question}</question>

                To do so, first find the sentence from the haystack that contains the answer (there is such a sentence, I promise!) and put it inside <most_relevant_sentence> XML tags. Then, put your answer in <answer> tags. Base your answer strictly on the context, without reference to outside information. Thank you.
                If you can't find the answer return the single word UNANSWERABLE.
                Assistant:
    '''

    OPENAI_TEMPLATE = '''
                You are a helpful AI bot that answers questions for a user. Keep your response short and direct.
                The following is a set of context and a question that will relate to the context. 
                #CONTEXT
                {context}
                #ENDCONTEXT

                #QUESTION
                {question} Don't give information outside the document or repeat your findings. If the
                information is not available in the context respond UNANSWERABLE.
                '''
    #{question} You are looking for a number from the context. Don't give information outside the document or repeat your findings

    RANDOM_NEEDLE_CITIES  = [
    'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
    'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
    'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
    'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
    'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
    'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
    'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
    'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
    'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
    'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
    'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
    'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta']

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        contexts = []
        #Evaluation of the model performance 
        #Uses Phoenix Evals
        nest_asyncio.apply() 
        if self.model_provider == "OpenAI":
            model = OpenAIModel(model_name="gpt-4-1106-preview")
            template =self.OPENAI_TEMPLATE
        else:
            model = LiteLLMModel(model_name="claude-2.1", temperature=0.0)
            template =self.ANTHROPIC_TEMPLATE

        full_context = self.read_context_files()
        for context_length in self.context_lengths:
            trim_context = self.encode_and_trim(full_context, context_length)
            for depth_percent in self.document_depth_percents:
                # Randomly selecting a city
                random_city = random.choice(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES)
                #Insert the needle 10o% of the time
                insert_needle = True
                needle_rnd_number = str(self.generate_random_number(self.rnd_number_digits))
                print("context length: " + str(context_length))
                print("depth_percent : " + str(depth_percent))
                results = self.create_contexts(needle_rnd_number, insert_needle, random_city, trim_context, context_length, depth_percent)
                contexts.append(results)
        df = pd.DataFrame(contexts)
        # The rails is used to search outputs for specific values and return a binary value
        # It will remove text such as ",,," or "..." and general strings from outputs
        # It answers needle_rnd_number or unanswerable or unparsable (if both or none exist in output)
        rail_map = [[row['needle_rnd_number'], "UNANSWERABLE"] for index, row in df.iterrows()]


        #This is the core of the Phoenix evaluation
        #It runs the model on every row of the dataframe
        #It looks for columns that are defined in the template question/context
        #The generation of the model, the output, is "cleaned" up by the rails
        #The rails are used to search for specific values in the output
        #The output is then classified as either needle_rnd_number, unanswerable, or unparsable
        #This runs a number of threads in parallel speeding up the generation/Evaluation process
        relevance_classifications = llm_classify(
            dataframe=df,
            template=template,
            model=model,
            rails=rail_map,
            verbose=True,
            #provide_explanation=True,
            concurrency=15,
            #Functions will not work for this evaluation as you will give the model the answer
            use_function_calling_if_available=False 
        )
        run_name = self.model_provider + "_" + str(self.context_lengths_num_intervals)  + "_" + str(self.document_depth_percent_intervals) 
        df['relevance_classifications'] = relevance_classifications
        df['score'] = df.apply(lambda row: self.check_row(row), axis=1)
        df.to_csv("save_results_" + run_name + "_.csv")
        self.generate_image(df, run_name)

        ###############
        # NEGATIVE TEST
        ###############``
        contexts = []
        #Run the negative test to see if the model gets the right answer when it is not in the context
        for context_length in self.context_lengths:
            trim_context = self.encode_and_trim(full_context, context_length)
            #There is no need of depth as needle is not inserted
            # Randomly selecting a city
            random_city = random.choice(LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES)
            #Randomly do not insert the needle 10% of the time
            insert_needle = False
            needle_rnd_number = str(self.generate_random_number(self.rnd_number_digits))
            print("context length: " + str(context_length))
            print("depth_percent : " + str(depth_percent))
            results = self.create_contexts(needle_rnd_number, insert_needle, random_city, trim_context, context_length, depth_percent)
            contexts.append(results)
        df = pd.DataFrame(contexts)
        rail_map = [[row['needle_rnd_number'], "UNANSWERABLE"] for index, row in df.iterrows()]
        relevance_classifications = llm_classify(
            dataframe=df,
            template=template,
            model=model,
            rails=rail_map,
            verbose=True,
            concurrency=15,
            use_function_calling_if_available=False 
        )
        print("Negative Test")
        percentage_unanswerable = (relevance_classifications['label'] == 'unanswerable').mean() * 100
        print(f"Percentage of 'unanswerable': {percentage_unanswerable:.2f}%")
        return contexts

    # Modify the check_row function to accept needle_number
    def check_row(self, row):
        if row['insert_needle']:
            return 1 if row['relevance_classifications'] == row['needle_rnd_number'] else 10
        else:
            return 1 if row['relevance_classifications'] == 'unanswerable' else 10

    def create_contexts(self, needle_rnd_number, insert_needle, random_city, trim_context, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        #if insert_needle is false then the needle is not inserted
        if not insert_needle:
            needle = " " #replace needle with a space
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(needle, trim_context, context_length, depth_percent)
        results = {
            'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : needle,
            'question' : question,
            'insert_needle' : insert_needle,
            'needle_rnd_number' : needle_rnd_number
         }
        return results

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

    def generate_context(self, needle, trim_context, context_length, depth_percent):
        # Insert your random statement according to your depth percent
        context = self.insert_needle(needle, trim_context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(needle)
        tokens_context = self.encode_text_to_tokens(context)

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
            period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context


    def get_context_length_in_tokens(self, context):
        if self.model_provider == "OpenAI":
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.enc.encode(context).ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider == "OpenAI":
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider == "OpenAI":
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    

    def generate_image(self, csv_df, run_name):

        # File path for saving the plot as a PNG file
        output_png_path = run_name + "_graph.png"  # Replace with your desired file path

        # Extracting the unique context lengths and depth percentages
        context_lengths = sorted(set(csv_df['context_length']))
        depth_percents = sorted(set(csv_df['depth_percent']))

        # Define the figure size and calculate the marker size
        fig_width = 20  # Width of the figure
        fig_height = 10  # Height of the figure
        marker_size = (fig_width / len(context_lengths)) * (fig_height / len(depth_percents)) * (72**2)  # 72 points per inch

        # Define the red and green colors in RGB format for the custom colormap
        red_rgb = (0.88, 0.22, 0.21)  # A shade of red
        green_rgb = (0.35, 0.70, 0.30)  # A shade of green

        # Create a custom colormap from red to green, reversed
        custom_cmap_reversed = LinearSegmentedColormap.from_list("custom_green_red", [green_rgb, red_rgb], N=256)

        # Normalization object for the colormap
        norm = Normalize(vmin=1, vmax=10)  # Assuming scores are in the range 1 to 10

        # Create a new figure for the scatter plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlabel('Context Length (# Tokens)')
        ax.set_ylabel('Placed Fact Document Depth (%)')

            # Adjust the tick positions
        x_tick_positions = np.arange(len(context_lengths)) + 0.5
        y_tick_positions = np.arange(len(depth_percents)) + 0.15

        ax.set_xticks(x_tick_positions)
        ax.set_yticks(y_tick_positions)

        # Reverse the depth_percents list
        depth_percents = sorted(set(csv_df['depth_percent']), reverse=True)
        # Adjust the position of the X-axis labels
        ax.set_xticklabels(context_lengths, ha='right', rotation=90, rotation_mode="anchor")
        # Set the limits of the y-axis to the exact range of your depth_percents
        ax.set_ylim(0 - 0.7, len(depth_percents) - 0.7)
        # Adjust the position of the Y-axis labels
        ax.set_yticklabels(depth_percents, va='top', rotation=0)
        plt.xticks(rotation=90)
        ax.grid(which='both', color='#404040', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Plot each data point with a lower zorder
        for index, row in csv_df.iterrows():
            context_length = row['context_length']
            depth_percent = row['depth_percent']
            score = row['score']
            context_index = context_lengths.index(context_length)
            # Calculate depth_index based on the reversed depth_percents list
            depth_index = depth_percents.index(depth_percent)
            # Plot the data point
            ax.scatter(context_index + 1, depth_index - 0.6, s=marker_size, c=[score], cmap=custom_cmap_reversed, norm=norm, marker='s', zorder=1)

        # Adjust grid to be above the scatter plot
        ax.set_axisbelow(False)
        # Add a color bar
        scalarmappable = plt.cm.ScalarMappable(cmap=custom_cmap_reversed, norm=norm)
        plt.colorbar(scalarmappable, ax=ax)
        plt.tight_layout()

        # Save the figure to the specified file path
        plt.savefig(output_png_path, format='png')

        # Display the plot (optional)
        plt.show()

    def get_results(self):
        return self.testing_results
    
    
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
        self.run_test()

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    #We added the option for the rail to be different on every row to this RC
    #random number for each generation of the context
    install_package("arize-phoenix==1.9.1rc2")
    #Runs Arize Phoenix Evaluation
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = LLMNeedleHaystackTester()

    ht.start_test()