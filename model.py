import keras
import numpy as np
from settings import MODELS_PATH

class ModelSpesification:

    def __init__(self, id_: int, default: str = None, tags: tuple = None):
        self.id = id_
        self.default = default
        self.tags = tuple(tags) if tags else ()

        self.name = f"model_{self.default}_{self.id}" \
            if self.default else f"model_{self.id}"

        self.model_dir = MODELS_PATH / self.name
        self.log_dir = self.model_dir / "logs"
        self.figures_dir = self.model_dir / "figures"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        file_path = (self.model_dir / "model.keras")
        if not file_path.exists():
            raise FileExistsError("Models doesn't exist")

        return keras.models.load_model(file_path, compile=False)

    @property
    def prediction(self):
        file_path = (self.model_dir / "prediction.npy")
        if not file_path.exists():
            raise FileExistsError("Evaluation on the test set doesn't exist")

        return np.load(file_path)

    def __getitem__(self, item):
        if item not in self.tags:
            raise KeyError(f"Tag '{item}' not found. Available tags: {self.tags}")

        return ModelSpesification(id_=self.id, default=item)


class ResidualUnit(object):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default, it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true uses full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default, it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default, it is false.
    activation_function: string, optional
        Keras activation function to be used. By default, 'relu'.
    References
    ----------
     [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
     [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with down sampling
        if downsample > 1:
            y = keras.layers.MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = keras.layers.Conv1D(self.n_filters_out, 1, padding='same',
                                    use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = keras.layers.Activation(self.activation_function)(x)
            x = keras.layers.BatchNormalization(center=False, scale=False)(x)
        else:
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = keras.layers.Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                                use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = keras.layers.Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = keras.layers.Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                                padding='same', use_bias=False,
                                kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = keras.layers.Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)
        else:
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Add()([x, y])  # Sum skip connection and main connection
            x = keras.layers.Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


def get_model(n_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = keras.layers.Input(shape=(4096, 12), dtype=np.float32, name='signal')
    x = signal
    x = keras.layers.Conv1D(64, kernel_size, padding='same', use_bias=False,
                            kernel_initializer=kernel_initializer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(n_classes, kernel_initializer=kernel_initializer)(x)
    diagn = keras.layers.Activation(last_layer, dtype=np.float32)(x)

    model = keras.models.Model(signal, diagn)
    return model


if __name__ == "__main__":
    default_model = get_model(6)
    default_model.summary()
