import argparse
from ast import parse
import json
import os
import time
import openai
from tqdm import tqdm

# import shortuuid
import asyncio
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:
        if('score:' in review):
            score = float(review.split('score:')[1].split('\n')[0])
        elif('Score:' in review):
            score = float(review.split('Score:')[1].strip('\n')[0])
        else:           
            logger.error(
                f"{e}\nContent: {review}\n" "You must manually fix the score pair."
            )
            score = -1
    
    return score

def find_error_items(alpaca_data,alpaca_data_cleaned_archive):
    alpaca_data_cleaned_archive_str = set([str(d) for d in alpaca_data_cleaned_archive])
    dirty_list = []
    for i, x in enumerate(alpaca_data):
        x = str(x)
        if(x not in alpaca_data_cleaned_archive_str):
            dirty_list.append(i)
    return dirty_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="the batch size to call the ChatGPT."
    )
    args = parser.parse_args()
    message_list = []
    # alpaca_data_cleaned_archive = json.load(open("./alpaca_data_cleaned_archive.json"))
    alpaca_data = json.load(open("./alpaca_data.json"))
    system_prompt = "We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following."


    '''
    rating according to the helpfulness
    '''
    # user_prompt = "Please rate according to the helpfulness of the response to the instruction and the input. Each assistant receives an score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    # dirty_list = find_error_items(alpaca_data, alpaca_data_cleaned_archive)
    '''
    rating according to the accuracy
    '''
    user_prompt = "Please rate according to the accuracy of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the accuracy. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    print(f"Alpaca data pairs: {len(alpaca_data)}")
    for i in range(len(alpaca_data)):
        # import pdb; pdb.set_trace()
        triplet = "###Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n".format_map(alpaca_data[i])
        eval_prompt = triplet + user_prompt
        message =[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": eval_prompt,
                    },
        ]
        message_list.append(message)
    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    batch_size = args.batch_size
    pbar = tqdm(total=len(message_list))
    while(i<len(message_list)):
        try:
            batch_predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list[i:i+batch_size],
                    model="gpt-3.5-turbo-0301",
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    top_p=1.0,
                )
            )
            predictions += batch_predictions
            i += batch_size
            wait_base = 10
            pbar.update(batch_size)
        except:
            retry += 1
            print("Batch error: ",i, i+10)
            print("retry number: ", retry)
            time.sleep(wait_base)
            wait_base = wait_base*2
    pbar.close()

    outputs = []
    for idx, prediction in tqdm(enumerate(predictions)):
        review = prediction['choices'][0]['message']['content']
        # score = parse_score(review)
        triplet = "###Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n".format_map(alpaca_data[idx])
        meta_output = {
                    "triplet":triplet,
                    "review":review
                }
        outputs.append(meta_output)
    with open(f"{args.output_review_file}", "w") as output_review_file:
            json.dump(outputs, output_review_file, indent=4)

# if __name__=="__main__":
#     review = "3.5\n\nThe response provides a basic understanding of the difference between individual and societal performance. However, it lacks depth and does not provide specific examples or analysis to support the comparison and contrast. The language used is clear and concise, but the response could benefit from more elaboration and explanation. Overall, the response is helpful to a certain extent, but could be improved with more detail and analysis."
#     print(parse_score(review))
