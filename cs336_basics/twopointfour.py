import regex as re
from collections import defaultdict

# Not used in the current implementation, but could be useful for tokenization.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Takes the path, VOCAB_SIZE IS FINAL SIZE, and list of special tokens
# Outputs the vocabulary and merges list.

# VOCAB_SIZE includes special tokens and merges
def train_bpe(input_path, vocab_size, special_tokens):
    # (GOOD) Intial vocabulary. STEP 1
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = '<|endoftext|>'.encode('utf-8')  # This is specifically for the twopointfour example.
    for i in range(257, 257+len(special_tokens)):
        vocab[i] = special_tokens[i-257].encode('utf-8')  # Add any additional special tokens to the vocabulary.
   
    # (GOOD) Read the input text file and convert it to a list. STEP 2
    with open(input_path, 'r') as f:
        text = f.read()

    # (GOOD) Split based on whitespace. STEP 2
    words = text.split()

    frequencies = defaultdict(int)

    # (GOOD) Count the frequency of each chunk of bytes in the text 
    for word in words:
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

vocab, merges = train_bpe('twopointfour.txt', 263, [])
print(vocab)
print(merges)
