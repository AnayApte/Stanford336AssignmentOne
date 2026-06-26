import regex as re
from collections import defaultdict

# First implement the test adapter on adapters.run_train_bpe in adapters.py

# Takes the path, VOCAB_SIZE IS FINAL SIZE, and list of special tokens
# Outputs the vocabulary and merges list.

# VOCAB_SIZE includes special tokens and merges
def train_bpe(input_path, vocab_size, special_tokens):
    
    # (GOOD) Intial vocabulary. STEP 1
    vocab = {i: bytes([i]) for i in range(256)}
    for i in range(256, 256+len(special_tokens)):
        vocab[i] = special_tokens[i-256].encode('utf-8')  # Add any additional special tokens to the vocabulary.
   
    # (GOOD) Read the input text file and convert it to a list. STEP 2
    with open(input_path, 'r') as f:
        text = f.read()

    # (GOOD) Chunks the text based on the special tokens. STEP 2
    special_pattern = '|'.join(re.escape(token) for token in special_tokens)
    if special_tokens:
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    # Define a frequencies map and a regex pattern to match words, numbers, and punctuation. STEP 2
    frequencies = defaultdict(int)
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # Go through each chunk. re.finditer lazily matches the pattern so convert to string and do the same way as before.
    for chunk in chunks:
        if not chunk:
            continue
        for match in re.finditer(PAT, chunk):
            word = match.group()
            word_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
            frequencies[word_tuple] += 1
    
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
