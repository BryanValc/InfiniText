import json
import random as r
import re
from collections import defaultdict

text_file = open("input/alice.txt", "r")
target_text = open("input/target.txt", "r")

def save_model(file):
    # crete a list of words out of the text, first splitting the text to remove spaces and line breaks
    words = file.read().split()

    #now we replace everything that is not a letter with an empty string, using regular expressions
    for i in range(len(words)):
        words[i] = re.sub('[^a-z]', '', words[i].lower())

    #now we create the dictionary that will hold the model weights
    model = defaultdict(lambda: defaultdict(int))

    #now we iterate through the words and add the next word to the model, this is basically assigning the weights
    for i in range(len(words) - 1):
        model[words[i]][words[i + 1]] += 1

    #here we store the weights in a json file
    with open('model.json', 'w') as outfile:
        json.dump(model, outfile)

def load_model(path):
    # load the json as a Python dictionary
    with open(path, 'r') as infile:
        model = json.load(infile)

    # Convert back to defaultdict
    model = defaultdict(lambda: defaultdict(int), model)

    for key in model:
        model[key] = defaultdict(int, model[key])

    return model

def print_model(model):
    for word, weights in model.items():
        print(f"{word}:")
        for following_word, weight in weights.items():
            print(f"\t{following_word}: {weight}")


save_model(text_file)
model = load_model('model.json')
print_model(model)

