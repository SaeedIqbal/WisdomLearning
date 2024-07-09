import tensorflow as tf
from tensorflow.keras.layers import Concatenate

class FeatureFusion(tf.keras.layers.Layer):
    def __init__(self, fusion_method='concat'):
        super(FeatureFusion, self).__init__()
        self.fusion_method = fusion_method

    def call(self, visual_features, clinical_features):
        if self.fusion_method == 'concat':
            combined_features = Concatenate()([visual_features, clinical_features])
        elif self.fusion_method == 'add':
            combined_features = visual_features + clinical_features
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        return combined_features

