import regex as re
from collections import defaultdict
import pickle

class Tokenizer:
    
    # vocab is a dictionary of of token IDs in ints to token bytes.
    # merges is a list of tuples of bytes representing BPE merges.
    # special_tokens is a list of strings representing special tokens.
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        if self.special_tokens:
            self.special_pattern = '|'.join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def merge(self, word_tuple, pair):
        i = 0
        new_word = []

        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                new_word.append(b''.join(pair))
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1

        return tuple(new_word)
    
    def encode(self, text):
        # Split the text into chunks based on the special tokens. If no special tokens, treat the entire text as a single chunk.
        if self.special_tokens:
            chunks = re.split("("+self.special_pattern+")", text)
        else:
            chunks = [text]
        
        # Define a regex pattern to match words, numbers, and punctuation.
        PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        token_ids = []

        # Iterate through each chunk of the text
        for chunk in chunks:
            if not chunk:
                continue
            # If a chunk is a special token, append its corresponding token ID to the list and continue to the next chunk.
            if chunk in self.special_tokens:
                token_ids.append(self.inverse_vocab[chunk.encode('utf-8')])
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group()
                word_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
                
                # Create pairs and check if the pair is in the merges dictionary. Break if no pairs in there.
                while True:
                    pairs = list(zip(word_tuple, word_tuple[1:]))
                    valid_pairs = [pair for pair in pairs if pair in self.merges_dict]
                    if not valid_pairs:
                        break
                    
                    # Take the min rank merge and pair them together.
                    new_token = min((pair for pair in valid_pairs), key=lambda p: self.merges_dict[p])
                    word_tuple = self.merge(word_tuple, new_token)
                
                for token in word_tuple:
                    token_ids.append(self.inverse_vocab[token])

        return token_ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids):
        string = b''.join(self.vocab[token_id] for token_id in ids).decode('utf-8', errors='replace')
        return string
