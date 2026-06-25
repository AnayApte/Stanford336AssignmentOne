import regex as re
from collections import defaultdict

# Not used in the current implementation, but could be useful for tokenization.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Start by creating a vocabulary of bytes,
# where the key is the index of the byte and the value is the byte itself.
vocab = {}
for i in range(256):
    vocab[i] = bytes([i])
vocab[256] = '<|endoftext|>'.encode('utf-8')
special_tokens = [256]  # List of special tokens, starting with the end-of-text token.

def train_bpe(input_path, vocab_size, special_tokens):
    # Read the input text file and convert it to a list.
    with open(input_path, 'r') as f:
        text = f.read()

    # Split based on whitespace.
    words = text.split()
    frequencies = defaultdict(int)

    # Count the frequency of each chunk of bytes in the text.
    for word in words:
        word_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
        frequencies[word_tuple] += 1
    
    # Creates a dictionary counting the frequency of byte pairs
    byte_pairs = defaultdict(int)
    for word, freq in frequencies.items():
        for i in range(len(word) - 1):
            byte_pairs[(word[i], word[i + 1])] += freq
    
    # Sort the byte pairs by frequency.
    byte_pairs = sorted(byte_pairs.items(), key=lambda x: (-x[1], x[0]))
    
    merges = []

    next_id = max(vocab.keys()) + 1  # Start assigning new IDs after the last existing ID.

    while len(merges) < vocab_size - len(vocab):
        if(len(byte_pairs) == 0):
            break
        merges.append(byte_pairs[0][0]) # Add the most frequent byte pair to the merges list.
        vocab[next_id] = b''.join(merges[-1]) # Add the new token to the vocabulary.
        next_id += 1

        new_frequencies = defaultdict(int)
        for word, freq in frequencies.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == merges[-1]:
                    new_word.append(b''.join(merges[-1]))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_frequencies[tuple(new_word)] += freq
        frequencies = new_frequencies
        byte_pairs = defaultdict(int)
        for word, freq in frequencies.items():
            for i in range(len(word) - 1):
                byte_pairs[(word[i], word[i + 1])] += freq
    
        # Sort the byte pairs by frequency.
        byte_pairs = sorted(byte_pairs.items(), key=lambda x: (-x[1], x[0]))

    return vocab, merges

vocab, merges = train_bpe('twopointfour.txt', 262, special_tokens)
print(vocab)
print(merges)
