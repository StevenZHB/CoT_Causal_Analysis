import re
import json
from tqdm import tqdm
import random
import os
import argparse

def extract_logic(answer):
    pattern1 = r"correct \w+ is:?\s*([A-D])"
    pattern2 = r"correct option is: (true|false|unknown)"
    pattern3 = r"([A-C])\)\s*(True|False|Unknown)"
    pattern4 = r"([A-D])\) "
    pattern5 = r"^[A-D]\.?$"

    match = re.search(pattern1, answer)
    option = None
    # extract pattern
    if match:
        option = match.group(1)
    
    if not option:
        match = re.search(pattern2, answer, re.IGNORECASE)
        if match:
            word_to_option = {"true": "A", "false": "B", "unknown": "C"}
            option = word_to_option.get(match.group(1).lower())

    if not option:
        match = re.search(pattern3, answer, re.IGNORECASE)
        if match:
            option = match.group(1)
    if not option and len(answer)<16:
        if 'true' in answer.lower():
            option = 'A'
        elif 'false' in answer.lower():
            option = 'B'
        elif 'unknown' in answer.lower():
            option = 'C'
    if not option:
        match = re.match(pattern4, answer)
        if match:
            option = match.group(1)
    if not option:
        match = re.match(pattern5, answer)
        if match:
            option = match.group(0) 
    if not option:
        option = None
        # wrong_data.append(d)
    return option


def human_check(sample):
    # Print the sample
    print(f"Found wrong data, please check the reasoning and extract the answer.")
    predicted_reasoning = sample.pop("predicted_reasoning")

    # Add 'predicted_reasoning' back to the dictionary, but it will be placed at the end
    sample["predicted_reasoning"] = predicted_reasoning
    print(sample)
    
    
    # Get user input
    mark = input("Extract the option(A/B/C): ")
    return mark


