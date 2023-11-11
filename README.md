# Pressure Testing GPT-4-128K

A simple 'needle in a haystack' analysis to test in-context retrieval ability of GPT-4-128K context

**The Test**
1. Place a random fact or statement (the 'needle') in the middle of a long context window
2. Ask the model to retrieve this statement
3. Iterate over various document depths (where the needle is placed) and context lengths to measure performance

This is the code that backed [this tweet](https://twitter.com/GregKamradt/status/1722386725635580292).

If ran, this script will populate `results.json` with evaluation information. Original results are held within `/original_results`, though they don't have as much information as they should. The current script gathers and saves more data.

The key pieces:

 * `needle` : The random fact or statement you'll place in your context
 * `question_to_ask`: The question you'll ask your model which will prompt it to find your needle/statement
*  `results_version`: Set to 1. If you'd like to run this test multiple times for more data points change this value to your version number
* `context_lengths` (List[int]): The list of various context lengths you'll test. In the original test this was set to 15 evenly spaced iterations between 1K and 128K (the max)
* `document_depth_percents` (List[int]): The list of various depths to place your random fact 
* `model_to_test`: The original test chose `gpt-4-1106-preview`. You can easily change this to any chat model from OpenAI, or any other model w/ a bit of code adjustments

## Results Visualization
![alt text](ResultsVisualization.png "Title")
(Made via pivoting the results, averaging the multiple runs, and adding labels in google slides)