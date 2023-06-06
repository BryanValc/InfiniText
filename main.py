import json
import random as r
import re
from collections import defaultdict

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



def encode_target_text(text):
    # in this method we create a 26 bit binary string that represents true or false for each letter of the alphabet in the target text
    # we will use this to discriminate words from the model that are not in the target text, for instance, if the text only contain the
    # letter a b and c, the first 3 bits of the string will be 1 and the rest will be 0

    # we will do the same with the lengths of the words, so we can discriminate words that are too long or too short, using a 16 bit binary string
    # for instance, if the target text only contains words of length 3 and, the binary string will be 0010100000000000
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    bit_string_letters = ['0'] * 26
    bit_string_length = ['0'] * 16
    bit_string_word_count = ['0'] * 8

    words = re.findall(r'\b\w+\b', text)
    unique_letters = set(text.lower())  # gets all unique letters in the text
    word_lengths = set(len(word) for word in words)  # gets all unique word lengths in the text
    word_count = len(words)

    # Check if the letter is in unique_letters
    for i, letter in enumerate(alphabet):
        if letter in unique_letters:
            bit_string_letters[i] = '1'

    # Check if the length is in word_lengths
    for length in word_lengths:
        if length <= 16:  # we only have space for lengths up to 16
            bit_string_length[length - 1] = '1'  # subtract 1 because indices start at 0

    # Convert the word_count to 8-bit binary and store it in bit_string_word_count
    if word_count <= 255:
        bit_string_word_count = list(format(word_count, '08b'))

    return ''.join(bit_string_letters), ''.join(bit_string_length), ''.join(bit_string_word_count)

def create_submodel(model, letters_bit_string, length_bit_string):
    submodel = defaultdict(lambda: defaultdict(int))

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    allowed_letters = {letter for i, letter in enumerate(alphabet) if letters_bit_string[i] == '1'}
    allowed_lengths = {i + 1 for i in range(16) if length_bit_string[i] == '1'} # indices are 0-based, lengths are 1-based

    for word, weights in model.items():
        if set(word).issubset(allowed_letters) and len(word) in allowed_lengths:
            for following_word, weight in weights.items():
                if set(following_word).issubset(allowed_letters) and len(following_word) in allowed_lengths:
                    submodel[word][following_word] = weight

    return submodel

def get_word_index(target_text, submodel):
    words = sorted(submodel.keys())
    print(words)
    word_to_index = {word: index for index, word in enumerate(words)}
    first_word = target_text.split()[0] if target_text.strip() != "" else None
    return format(word_to_index.get(first_word, 0), '016b')


text_file = open("input/alice.txt", "r")
target_text = open("input/target.txt", "r")

save_model(text_file)
model = load_model('model.json')
# print_model(model)

# encode the target text for discrimination of words
bit_string_letters, bit_string_length, bit_word_count = encode_target_text(target_text.read())
print(f"Letters bit string: {bit_string_letters}")
print(f"Length bit string: {bit_string_length}")
print(f"Word count bit string: {bit_word_count}")

# create a submodel based on the target text discrimination parameters
submodel = create_submodel(model, bit_string_letters, bit_string_length)
print_model(submodel)

# get the index of the first word in the target text
target_text = open("input/target.txt", "r")
word_index = get_word_index(target_text.read(), submodel)

print(f"Word index: {word_index}")

