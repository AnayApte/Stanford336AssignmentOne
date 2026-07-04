from cs336_basics.twopointfive import train_bpe
if __name__ == "__main__":
    vocab, merges = train_bpe('/Users/anayapte/Desktop/Stanford CS336/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
    print(f'Vocab size: {len(vocab)}')
    print(f'Merges size: {len(merges)}')
    print(f'Longest token: {max(vocab.values(), key=len)}')
