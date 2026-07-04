import regex as re
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
import os

# First implement the test adapter on adapters.run_train_bpe in adapters.py

# Takes the path, VOCAB_SIZE IS FINAL SIZE, and list of special tokens
# Outputs the vocabulary and merges list.

# VOCAB_SIZE includes special tokens and merges
def process_chunk(args):
    # Unpack the arguments, must do this because multiprocessing.Pool.map only takes a single argument.
    input_path, start, end, special_tokens = args

    # Read the inputted chunk of the file, taken as bytes because of find_chunk_boundaries.
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode('utf-8', errors='ignore')

    # Split the text into chunks based on the special tokens. If no special tokens, treat the entire text as a single chunk.
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    if special_tokens:
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]
    
    # Define a frequencies map and a regex pattern to match words, numbers, and punctuation.
    frequencies = defaultdict(int)
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    for chunk in chunks:
        if not chunk:
            continue
        for match in re.finditer(PAT, chunk):
            word = match.group()
            word_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
            frequencies[word_tuple] += 1

    return frequencies

def train_bpe(input_path, vocab_size, special_tokens):
    
    # (GOOD) Intial vocabulary. STEP 1
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(256, 256+len(special_tokens)):
        vocab[i] = special_tokens[i-256].encode('utf-8')  # Add any additional special tokens to the vocabulary.

    # Num Processes can be found by os.cpu_count() which is # of CPU cores on machine.
    num_processes = os.cpu_count()

    # Find the chunk boundaries as a list, special_tokens[0] is usually the first to delimit
    boundaries = find_chunk_boundaries(open(input_path, "rb"), num_processes, b"<|endoftext|>")

    # How to run on multiple cores, the argument it takes is a tuple of the input path, start and end of the chunk, and the special tokens. The function returns a list of frequency dictionaries for each chunk.
    with Pool(num_processes) as pool:
        frequencySet = pool.map(process_chunk, [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])

    # Combine the frequencies from all chunks into a single dictionary
    frequencies = defaultdict(int)
    for f in frequencySet:
        for key, count in f.items():
            frequencies[key] += count

    merges = []
    initialSize = len(vocab)

    while len(merges) < vocab_size - initialSize:

        # Byte pairs will keep changing based on words. Need to create a new one each loop.
        bytePairs = defaultdict(int)
        for word, freq in frequencies.items():
            for i in range(len(word) - 1):
                bytePairs[(word[i], word[i + 1])] += freq
        
        # Case where all possible pairs merged - ex: vocab_size is so large
        if(len(bytePairs) == 0):
            break

        # Sort the byte pairs by frequency. Break by lexicographical order.
        bytePairs = max(bytePairs.items(), key=lambda x: (x[1], x[0]))

        merges.append(bytePairs[0]) # Add the most frequent byte pair to the merges list.
        vocab[len(vocab)] = (b''.join(merges[-1])) # Add the new merged byte pair to the vocabulary.

        for word_bytes in list(frequencies.keys()):
            i = 0
            new_word = []
            while i < len(word_bytes) - 1:
                if word_bytes[i: i+2] == merges[-1]:
                    new_word.append(b''.join(merges[-1]))
                    i += 2
                else:
                    new_word.append(word_bytes[i])
                    i += 1
            if i == len(word_bytes) - 1:
                new_word.append(word_bytes[i])
            freq = frequencies.pop(word_bytes)
            frequencies[tuple(new_word)] += freq

    return vocab, merges
