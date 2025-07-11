import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizertest import TokenizerTest

def main():
    tokenizer_test = TokenizerTest()
    tokenizer_test.run_tests()

if __name__ == "__main__":
        main()