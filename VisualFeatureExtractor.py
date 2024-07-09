import tensorflow as tf
'''
This module implements the VisualFeatureExtractor class for extracting visual features from medical images using CNNs.

Author: [Saeed Iqbal]
Date: [December 05, 2023]

Usage:
    # Example usage of VisualFeatureExtractor
    visual_extractor = VisualFeatureExtractor()
    features = visual_extractor.extract_features(image_data)
    print(features)

Notes:
    This implementation provides a flexible framework for extracting visual features using Convolutional Neural Networks (CNNs).
'''
class VisualFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        
        # Separable Convolution layers
        self.sep_conv1 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')
        self.sep_conv2 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')
        
        # Combination of Conv2D and Separable Convolution
        self.combined_conv = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.combined_sep_conv = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        combined_features = self.combined_conv(inputs) + self.combined_sep_conv(inputs)
        x = tf.keras.layers.Concatenate()([x, combined_features])
        
        return x

# Example usage:
# Initialize and use VisualFeatureExtractor within your model architecture
visual_extractor = VisualFeatureExtractor()
input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
output_tensor = visual_extractor(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

# Print model summary
model.summary()

