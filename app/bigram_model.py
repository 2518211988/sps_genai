import random
from collections import defaultdict


class BigramModel:
    """A simple bigram language model."""

    def __init__(self, corpus):
        self.bigrams = defaultdict(list)
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                self.bigrams[current_word].append(next_word)

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        result = [current_word]
        for _ in range(length - 1):
            next_words = self.bigrams.get(current_word)
            if not next_words:
                break
            current_word = random.choice(next_words)
            result.append(current_word)
        return " ".join(result)
