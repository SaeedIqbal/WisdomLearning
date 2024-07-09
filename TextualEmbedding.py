import pickle

class TextualEmbedding:
    def __init__(self, data, word_count_threshold=10):
        self.data = data
        self.word_count_threshold = word_count_threshold
        self.word_counts = {}
        self.vocab = []
        self.wordtoix = {}
        self.ixtoword = {}
        self.vocab_size = 0
        self.max_seq_length = 0
        
        self._initialize()

    def _initialize(self):
        self.all_train_captions = self._extract_captions()
        self._build_vocabulary()
        self._convert_words_to_indices()
        self._save_word_mappings()
        self.max_seq_length = self._calculate_max_length()

    def _extract_captions(self):
        all_train_captions = []
        for key, val in self.data.items():
            for cap in val:
                all_train_captions.append(cap)
        return all_train_captions

    def _build_vocabulary(self):
        for sent in self.all_train_captions:
            for w in sent.split(' '):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1

        self.vocab = [w for w in self.word_counts if self.word_counts[w] >= self.word_count_threshold]
        print(f'Preprocessed words {len(self.word_counts)} -> {len(self.vocab)}')

    def _convert_words_to_indices(self):
        ix = 1
        for w in self.vocab:
            self.wordtoix[w] = ix
            self.ixtoword[ix] = w
            ix += 1
        self.vocab_size = len(self.ixtoword) + 1  # one for appended 0's

    def _save_word_mappings(self):
        with open("clinical.pkl", "wb") as encoded_pickle:
            pickle.dump(self.wordtoix, encoded_pickle)
            
        with open("clinicalN.pkl", "wb") as encoded_pickle:
            pickle.dump(self.ixtoword, encoded_pickle)

    def to_lines(self, descriptions):
        all_desc = []
        for key in descriptions.keys():
            all_desc.extend(descriptions[key])
        return all_desc

    def _calculate_max_length(self):
        lines = self.to_lines(self.data)
        return max(len(d.split()) for d in lines)

    def get_vocab_size(self):
        return self.vocab_size

    def get_max_seq_length(self):
        return self.max_seq_length


# Example usage
data2 = cleanse_data(data)
text_embedder = TextualEmbedding(data2)
