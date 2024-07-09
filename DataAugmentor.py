import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DataAugmentor:
    def __init__(self, descriptions, photo_features, word_to_index, max_seq_length, vocab_size, batch_size):
        self.descriptions = descriptions
        self.photo_features = photo_features
        self.word_to_index = word_to_index
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def generate_data(self):
        X1, X2, y = [], [], []
        n = 0
        while True:
            for key, desc_list in self.descriptions.items():
                n += 1
                photo = self.photo_features[key]
                for desc in desc_list:
                    seq = self._encode_sequence(desc)
                    for i in range(1, len(seq)):
                        in_seq, out_seq = self._create_input_output_pair(seq, i)
                        X1.append(photo)
                        X2.append(in_seq)
                        y.append(out_seq)
                if n == self.batch_size:
                    yield ([np.array(X1), np.array(X2)], np.array(y))
                    X1, X2, y = [], [], []
                    n = 0

    def _encode_sequence(self, desc):
        return [self.word_to_index[word] for word in desc.split(' ') if word in self.word_to_index]

    def _create_input_output_pair(self, seq, index):
        in_seq = seq[:index]
        out_seq = seq[index]
        in_seq = pad_sequences([in_seq], maxlen=self.max_seq_length)[0]
        out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
        return in_seq, out_seq

# Example usage
data2 = cleanse_data(data)
text_embedder = TextualEmbedding(data2)
wordtoix = text_embedder.wordtoix
vocab_size = text_embedder.get_vocab_size()
max_length = text_embedder.get_max_seq_length()

photo_features = {}  # Load or define your photo features dictionary
descriptions = {}    # Load or define your descriptions dictionary

data_augmentor = DataAugmentor(descriptions, photo_features, wordtoix, max_length, vocab_size, num_photos_per_batch=32)
data_generator = data_augmentor.generate_data()

