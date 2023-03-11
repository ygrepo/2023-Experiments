import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# def reparam_trick(z_mean, z_log_var):
#     batch = tf.shape(z_mean)[0]
#     dim = tf.shape(z_mean)[1]
#     epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# def Encoder(enc_input_dim, enc_output_dim, enc_hidden_dims):

#     latent_dim = enc_output_dim

#     enc_inputs = Input(shape=(enc_input_dim,), name='encoder_input')
#     enc_hidden = [Dense(dim, activation='linear',
#             name='enc_hidden_{}'.format(i)) for i, dim in enumerate(enc_hidden_dims)]

#     z_mean_layer = Dense(latent_dim, activation='linear', name='z_mean')
#     z_log_var_layer = Dense(latent_dim, activation='linear', name='z_log_var')

#     # encoder evaluation
#     x = enc_inputs
#     for layer in enc_hidden:
#         x = ELU(layer(x))

#     z_mean = z_mean_layer(x)
#     z_log_var = z_log_var_layer(x)

#     #----- REPARAM TRICK ---------
#     z = reparam_trick(z_mean, z_log_var)


#     # instantiate encoder model
#     encoder = Model(enc_inputs, [z_mean, z_log_var, z], name='encoder')

#     return encoder, enc_inputs


class Encoder(layers.Layer):
    def __init__(self, latent_dim=2, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z


# def Decoder(dec_input_dim, dec_output_dim, dec_hidden_dims):

#     latent_dim = dec_input_dim

#     latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
#     dec_hidden = [Dense(dim, activation='linear',
#             name='dec_hidden_{}'.format(i)) for i, dim in enumerate(dec_hidden_dims)]

#     dec_output_layer = Dense(dec_output_dim, activation='linear')

#     # decoder evaluation
#     x = latent_inputs
#     for layer in dec_hidden:
#         x = ELU(layer(x))

#     dec_outputs = dec_output_layer(x)

#     # instantiate decoder model
#     decoder = Model(latent_inputs, dec_outputs, name='decoder')

#     return decoder


class Decoder(layers.Layer):
    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


# encoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def train():
    x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=30, batch_size=128)    