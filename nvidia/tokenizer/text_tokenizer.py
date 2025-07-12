class Tokenizer: 
    def __init__(self, vocab_size): 
        self.vocab_size = vocab_size
        self.vocab = {}
    
    def tokenize(self, text):
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

            id = self.vocab[word]
            tokens.append(id)

        return tokens

    def detokenize(self, tokens):
        text = ""
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        for token in tokens:
            if token in values:
                text += keys[values.index(token)] + " "
            else:
                text += "<UNK>" + " "

        return text.strip()
    
    def fill_dictionary(self, list_of_strings):
        for string in list_of_strings:
            self.tokenize(string)

        return self.vocab 