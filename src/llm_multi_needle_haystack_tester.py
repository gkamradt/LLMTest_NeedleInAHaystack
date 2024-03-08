from .llm_needle_haystack_tester import LLMNeedleHaystackTester

class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    """
    Extends LLMNeedleHaystackTester to support testing with multiple needles in the haystack.
    
    Attributes:
        needles (list): A list of needles (facts) to insert into the haystack (context).
        eval_set (str): The evaluation set identifier.
    """
    def __init__(self,
                 needles=[],
                 eval_set = "multi-needle-eval-sf",
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.needles = needles
        self.eval_set = eval_set

    async def insert_needles(self, context, depth_percent, context_length):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
        that the total token count (context plus needles) does not exceed the maximum allowable context length, 
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even 
        spacing for the remaining needles based on the remaining context length. It ensures that needles are 
        distributed as evenly as possible throughout the context after the first insertion. 
        
        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.
        
        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # To evenly distribute the needles, we calculate the intervals they need to be inserted.
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        # Insert needles at calculated points
        for needle in self.needles:
            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)
            # Insert each needle at its corresponding depth percentage
            # For simplicity, evenly distribute needles throughout the context
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
            # Adjust depth for next needle
            depth_percent += depth_percent_interval  

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    async def generate_context(self, context_length, depth_percent):
        """
        Generates a context of a specified length and inserts needles at given depth percentages.
        
        Args:
            context_length (int): The total length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        
        Returns:
            str: The context with needles inserted.
        """
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = await self.insert_needles(context, depth_percent, context_length)
        return context
    
    async def evaluate_and_log(self, context_length, depth_percent):
        """
        Evaluates the model's performance with the generated context and logs the results.
        
        Args:
            context_length (int): The length of the context in tokens.
            depth_percent (float): The depth percent for needle insertion.
        """
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent)

        # LangSmith
        ## TODO: Support for many evaluators 
        if self.evaluation_model.__class__.__name__ == "LangSmithEvaluator":
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluation_model.evaluate_chain(chain, context_length, depth_percent, self.model_name, self.eval_set)
        else:
            await super().evaluate_and_log(context, context_length, depth_percent)

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needles In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needles: {[needle.strip() for needle in self.needles]}")
        print ("\n\n")