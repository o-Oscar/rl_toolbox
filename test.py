import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

input = layers.Input(shape=(None, 3))
output = input*3

primitive = tf.keras.Model(input, output, name="actor")

m_input = layers.Input(shape=(None, 3))
mean = m_input * 2

mean = primitive(m_input)

m_output = mean*5

model = tf.keras.Model(m_input, m_output, name="actor")
help(model)
model.summary()
print(model(np.ones((4,3), dtype=np.float32)))