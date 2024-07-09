import numpy as np

class ClinicalFeatureMatrix:
    def __init__(self, embedding_dim=200, glove_path='glove.6B.200d.txt'):
        self.embedding_dim = embedding_dim
        self.glove_path = glove_path
        self.embeddings_index = self._load_glove_embeddings()

    def _load_glove_embeddings(self):
        embeddings_index = {}
        with open(self.glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def create_embedding_matrix(self, vocab_size, word_to_index):
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        for word, i in word_to_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

# Example usage
text_embedder = TextualEmbedding(data2)
word_to_index = text_embedder.wordtoix
vocab_size = text_embedder.get_vocab_size()

clinical_feature_matrix = ClinicalFeatureMatrix(embedding_dim=200, glove_path='/home/ph.d/dataset/glove.6B.200d.txt')
embedding_matrix = clinical_feature_matrix.create_embedding_matrix(vocab_size, word_to_index)
print(embedding_matrix.shape)

