import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# define the model architecture

class Model:
    def __init__(self, lr, x_train, y_train):
        inputs = keras.Input(shape=(28, 28, 1), name="digits")
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPool2D()(conv1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPool2D()(conv2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPool2D()(conv3)
        flatten = layers.Flatten()(pool3)
        x1 = layers.Dense(64, activation="relu")(flatten)
        outputs = layers.Dense(10, name="predictions")(x1)
        self.x_train = x_train
        self.y_train = y_train
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.SGD(learning_rate=1)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.shapes = []
        self.flat_gradient_shape = []
        self.calculate_gradients(x_train[0:1], y_train[0:1])
        self.accuracy = []
        self.loss = []

    def calculate_gradients(self, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = self.model(x_train, training=True)
            loss_value = self.loss_fn(y_train, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights, )
        result = self.flatten_gradients(grads)
        self.flat_gradient_shape = result.numpy().shape
        return self.flatten_gradients(grads)

    def flatten_gradients(self, gradients):
        flat_grad = []
        shapes = []
        for arr in gradients:
            flat_grad.append(tf.reshape(arr, [-1, 1]))
            shapes.append(tf.shape(arr))
        self.shapes = shapes
        return tf.concat(flat_grad, axis=0)

    def unflatten(self, flat_grad):
        output = []
        cntr = 0
        for shape in self.shapes:
            num_elements = tf.math.reduce_prod(shape)
            params = tf.reshape(flat_grad[cntr:cntr + num_elements, 0], shape)
            params = tf.cast(params, tf.float32)
            cntr += num_elements
            output.append(params)
        return output

    def update_params(self, flat_grad):
        output = self.unflatten(flat_grad)
        self.optimizer.apply_gradients(zip(output, self.model.trainable_weights))
        acc, loss = self.report_performance(self.x_train, self.y_train)
        self.accuracy.append(acc)
        self.loss.append(loss)

    def report_performance(self, x_test, y_test):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        test_idx = np.random.permutation(len(x_test))
        test_batch_idx = np.array_split(test_idx, 60)
        for batchIdx in test_batch_idx:
            logits = self.model(x_test[batchIdx], training=False)
            lossValue = self.loss_fn(y_test[batchIdx], logits)/len(batchIdx)
            test_accuracy.update_state(y_test[batchIdx], logits)
            test_loss.update_state(lossValue)
        return test_accuracy.result().numpy(), test_loss.result().numpy()
