import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.tokenizer import Tokenizer

class TokenizerTest():
    def run_tests(self):
        self.test_tokenize()
        self.test_detokenize()
        self.test_fill_dictionary()

    def test_tokenize(self):
        "it should tokenize on word count"
        tokenizer = Tokenizer(100)
        text = "Tokenize this sentence"
        tokens = tokenizer.tokenize(text)
        # it should be 3 tokens on word token split
        assert len(tokens) == 3
        text = "You're a towel"
        # it should be 3 tokens on word token split
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 3

        # dictionary should be 6 tokens
        assert len(tokenizer.vocab) == 6

        print("Tokenize test passed")

    def test_detokenize(self):
        "it should detokenize on word count"
        tokenizer = Tokenizer(100)
        text = "Tokenize this sentence"
        tokens = tokenizer.tokenize(text)
        detokenized_text = tokenizer.detokenize(tokens)
        assert detokenized_text == text
        print("Detokenize test passed")

    def test_fill_dictionary(self):
        "it should fill the dictionary with the list of words"
        tokenizer = Tokenizer(100)
        list_of_strings = ["a", "b", "c", "d", "e", "f", "g"]
        tokenizer.fill_dictionary(list_of_strings)
        assert len(tokenizer.vocab) == 7 

        # it should use previous words if used before
        list_of_strings = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        tokenizer.fill_dictionary(list_of_strings)
        assert len(tokenizer.vocab) == 11
        print("Fill dictionary test passed")