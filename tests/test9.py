import tensorflow as tf

class SimilarityTensor():
    def __init__(self, tensor):
        self.tensor = tensor
        self.realMin = tf.math.reduce_min(self.tensor)
        self.realMax = tf.math.reduce_max(self.tensor)
    
    def adjust_range(self, newMin, newMax, margin=0.0):

        newRange = (newMax - newMin) - 2*margin

        self.tensor = self.tensor - tf.math.reduce_min(self.tensor)
        # Current min value is 0

        self.tensor = self.tensor / tf.math.reduce_max(self.tensor)
        # Current max value is 1
        self.tensor = self.tensor * newRange
        # Current max value is newRange

        self.tensor = self.tensor + newMin + margin
        # Current min value is newMin + margin
        # Current max value is newMax - margin
    
    def reset_range(self):
        self.adjust_range(self.realMin, self.realMax)

a = tf.Variable([[5, 8, 1],
                 [1, 3, 4],
                 [6, 5, 2],
                 [7, 9, 1]], dtype=tf.float32)

b = a * 1
c = SimilarityTensor(b)
print("breakpoint")

b = b * -1

print("breakpoint")