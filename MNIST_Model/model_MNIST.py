
from keras import layers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding3D
from keras.layers.core import Activation, Dense, Reshape, Flatten
from keras.layers.merge import Add
from keras.models import Model
from keras.initializers import Constant, glorot_uniform
from keras import backend
import numpy as np

class UCC_Model_MNIST(object):

    # defining models
    #---------------------------------------------------------------------------
    def __init__(self, image_size=28, num_images=2, num_classes=10, learning_rate=0.001, num_KDE_bins=None, encoded_size=10, batch_size=None):
        """UCC Model for MNIST data clustering"""

        # building the encoder
        #-----------------------------------------------------------------------
        image_input = Input((image_size, image_size, 1))
        x_enc = Conv2D(16, (3, 3), padding="same", bias_initializer=Constant(0.1), kernel_initializer=glorot_uniform(), kernel_regularizer=None)(image_input)
        x_enc = self.residual_layer(x_enc, 2, 1, 16, sample=False, reverse=False)
        x_enc = self.residual_layer(x_enc, 4, 1, 16, sample=True, reverse=False)
        x_enc = self.residual_layer(x_enc, 8, 1, 16, sample=True, reverse=False)
        x_enc = Activation("relu")(x_enc)
        x_enc = Flatten()(x_enc)
        encoded_output = Dense(encoded_size, activation="sigmoid", use_bias=False, kernel_initializer=glorot_uniform(), kernel_regularizer=None)(x_enc)
        self._encoder_model = Model(inputs=image_input, outputs=encoded_output)


        # building the decoder
        #-----------------------------------------------------------------------
        encoded_input = Input((encoded_size,))
        x_dec = Dense(6272, bias_initializer=Constant(0.1), kernel_initializer=glorot_uniform(), kernel_regularizer=None)(encoded_input)
        x_dec = Reshape((7, 7, 128))(x_dec)
        x_dec = self.residual_layer(x_dec, 8, 1, 16, sample=True, reverse=True)
        x_dec = self.residual_layer(x_dec, 4, 1, 16, sample=True, reverse=True)
        x_dec = self.residual_layer(x_dec, 2, 1, 16, sample=False, reverse=True)
        x_dec = Activation("relu")(x_dec)
        decoded_output = Conv2D(1, (3, 3), padding="same", bias_initializer=Constant(0.1), kernel_initializer=glorot_uniform(), kernel_regularizer=None)(x_dec)
        self._decoder_model = Model(inputs=encoded_input, outputs=decoded_output)


        # build total autoencoder model
        #-----------------------------------------------------------------------
        autoencoded_output = self._decoder_model(encoded_output)
        self._autoencoder_model = Model(inputs=image_input, outputs=autoencoded_output)


        # running the encoder/autoencoder models to get out img and feature list
        #-----------------------------------------------------------------------
        original_input_list = []
        encoded_output_list = []
        autoencoded_output_list = []
        for i in range(num_images):
            # get input
            input_tmp = Input((image_size, image_size, 1))
            original_input_list.append(input_tmp) # add input image to list

            # Run encoder model
            output_tmp = self._encoder_model(input_tmp)
            output_tmp = Reshape((1, -1))(output_tmp)
            encoded_output_list.append(output_tmp) # add encoded image to list

            # Run autoencoder model
            autoencoded_output_tmp = self._autoencoder_model(input_tmp)
            autoencoded_output_tmp = Reshape((1, image_size, image_size, 1))(autoencoded_output_tmp)
            autoencoded_output_list.append(autoencoded_output_tmp) # add autoencoded image to list


        # build kernel density estimation model
        #-----------------------------------------------------------------------
        concatenated_encoded = layers.concatenate(encoded_output_list, axis=1)
        concatenated_autoencoded = layers.concatenate(autoencoded_output_list, axis=1)
        x_kde = layers.Lambda(self.kernel_density_estimation, arguments={"num_bins":num_KDE_bins, "sigma":0.1, "batch_size":batch_size, "num_features":encoded_size})(concatenated_encoded)
        self._distribution_model = Model(inputs=original_input_list, outputs=x_kde)


        # build classification model
        #-----------------------------------------------------------------------

        x_class = Dense(384, activation="relu", kernel_initializer=glorot_uniform(), kernel_regularizer=None)(x_kde)
        x_class = Dense(192, activation="relu", kernel_initializer=glorot_uniform(), kernel_regularizer=None)(x_class)
        class_output = Dense(num_classes, activation="softmax", kernel_initializer=glorot_uniform(), kernel_regularizer=None)(x_class)
        self._classification_model = Model(inputs=original_input_list, outputs=[class_output, concatenated_autoencoded])


        # define some more models (might be able to delete some of these)
        #-----------------------------------------------------------------------
        self._features_model = Model(inputs=original_input_list, outputs=concatenated_encoded)
        self._ucc_model = Model(inputs=original_input_list, outputs=class_output)


        # set optimiser and loss function
        #-----------------------------------------------------------------------
        optimiser = Adam(lr = learning_rate)
        self._classification_model.compile(optimizer=optimiser, loss=["categorical_crossentropy", "mse"], metrics=["accuracy"], loss_weights=[0.5, 0.5])

    
    # defining kde function for kde layer
    # --------------------------------------------------------------------------
    def kernel_density_estimation(self, features, num_bins=None, sigma=None, batch_size=None, num_features=None):
        """kernel density esimation function used in the kde layer
        
        For some reason, this function does not work if we use some numpy functions
        like np.sum, or Keras layers like Flatten and Reshape. So instead we
        have to get these from keras.backends. I have tried to fix this, but I
        cannot, so leaving in for a future person to solve! """

        x = np.linspace(0, 1, num_bins)
        sample_points = backend.constant(np.tile(x, [batch_size, backend.int_shape(features)[1], 1]))

        a = backend.constant((np.sqrt(2 * np.pi * sigma**2)**-1))
        b = backend.constant((-2 * sigma**2)**-1)

        output_list = []
        for i in range(num_features):
            tmp = backend.reshape(features[:,:,i], (-1, backend.int_shape(features)[1], 1))
            diff = (sample_points - backend.tile(tmp, [1, 1, num_bins]))**2
            result = a * backend.exp(b * diff)

            output = backend.sum(result, axis=1)

            normalisaton_coefficient = backend.reshape(backend.sum(output, axis=1), (-1,1))

            output_normed = output / backend.tile(normalisaton_coefficient, [1, backend.int_shape(output)[1]])
            output_list.append(output_normed)

        concatenated_output = backend.concatenate(output_list, axis=-1)
        return concatenated_output


    # defining residual layer used in autoencoding model
    # --------------------------------------------------------------------------
    def residual_layer(self, x, filters, n, channels, sample=False, reverse=False):
        """residual layer definition
        
        if the reverse parameter is set to True, it executes the reverse"""

        for i in range(n):
            if i == 0:
                x = self.residual_block(x, filters * channels, first=True, sample=sample, reverse=reverse)
            else:
                x = self.residual_block(x, filters * channels, first=False, sample=sample, reverse=reverse)

        return x


    # defining residual block defined in residual layer
    # --------------------------------------------------------------------------
    def residual_block(self, x0, filters, first=False, sample=False, reverse=False):
        
        kwargs_1 = {"kernel_size": (3, 3),
                    "padding": "same",
                    "bias_initializer": Constant(value=0.1),
                    "use_bias": True,
                    "kernel_initializer": glorot_uniform(),
                    "kernel_regularizer": None
	}
        kwargs_2 = {"kernel_size": (1, 1),
                    "padding": "valid",
                    "use_bias": True,
                    "kernel_initializer": glorot_uniform(),
                    "kernel_regularizer": None}

        if first:
            if reverse == False:
                    # set appropriate strides to reduce size (downsample)
                    if sample == True:
                        kwargs_1["strides"] = (2, 2)
                        kwargs_2["strides"] = (2, 2)

                    x0 = Activation("relu")(x0)

                    x1 = Conv2D(filters, **kwargs_1)(x0)
                    x1 = Activation("relu")(x1)

                    # reset the stride length
                    kwargs_1["strides"] = (1, 1)

                    x1 = Conv2D(filters, **kwargs_1)(x1)

                    x0_shape = x0.shape.as_list()[1:-1]
                    x1_shape = x1.shape.as_list()[1:-1]
                    x0_filter = x0.shape.as_list()[-1]
                    x1_filter = x1.shape.as_list()[-1]

                    if x0_shape != x1_shape:
                            x0 = Conv2D(x0_filter, **kwargs_2)(x0)

                    if x0_filter != x1_filter:
                            t_shape = (x1_shape[0], x1_shape[1], x0_filter, 1)
                            x0 = Reshape(t_shape)(x0)
                            padding_size = x1_filter - x0_filter
                            x0 = ZeroPadding3D(((0, 0), (0, 0), (0, padding_size)))(x0)
                            t_shape = (x1_shape[0], x1_shape[1], x1_filter)
                            x0 = Reshape(t_shape)(x0)

            elif reverse == True:
                x0 = Activation("relu")(x0)

                # do appropriate step to increase size (upsampling)
                if sample == True:
                    x0 = UpSampling2D((2, 2))(x0)

                x1 = Conv2D(filters, **kwargs_1)(x0)
                x1 = Activation("relu")(x1)
                x1 = Conv2D(filters, **kwargs_1)(x1)

                if x0.shape.as_list()[-1] != x1.shape.as_list()[-1]:
                    x0 = Conv2D(filters, **kwargs_2)(x0)

        else:
            x1 = Activation("relu")(x0)
            x1 = Conv2D(filters, **kwargs_1)(x1)
            x1 = Activation("relu")(x1)
            x1 = Conv2D(filters, **kwargs_1)(x1)

        x0 = Add()([x0, x1])
        return x0


    # functions to train and test model
    # --------------------------------------------------------------------------

    def train_on_batch_data(self, batch_inputs=None, batch_outputs=None):
        stats = self._classification_model.train_on_batch(batch_inputs, batch_outputs)
        
        return stats

    def test_on_batch_data(self, batch_inputs=None, batch_outputs=None):
        stats = self._classification_model.test_on_batch(batch_inputs, batch_outputs)
        
        return stats

    def predict_on_batch_data(self, batch_inputs=None):
        predicted_label = self._classification_model.predict_on_batch(batch_inputs)
        
        return predicted_label

    def predict_ucc_on_batch_data(self, batch_inputs=None):
        predicted_label = self._ucc_model.predict_on_batch(batch_inputs)
        
        return predicted_label

    def predict_on_batch_data_ae(self, batch_inputs=None):
        predicted_label = self._autoencoder_model.predict_on_batch(batch_inputs)
        
        return predicted_label

    def generate_image_from_feature(self, batch_inputs=None):
        predicted_label = self._decoder_model.predict_on_batch(batch_inputs)
        
        return predicted_label

    def predict_on_batch_data_distribution(self, batch_inputs=None):
        predicted_dist = self._distribution_model.predict_on_batch(batch_inputs)
        return predicted_dist

    def predict_on_batch_data_features(self, batch_inputs=None):
        predicted_features = self._features_model.predict_on_batch(batch_inputs)
    
        return predicted_features

    def predict_on_batch_data_patches(self, batch_inputs=None):
        predicted_patches = self._encoder_model.predict_on_batch(batch_inputs)
        
        return predicted_patches

    def save_model_weights(self, model_weight_save_path=None):
        self._classification_model.save_weights(model_weight_save_path)
        self._autoencoder_model.save_weights(model_weight_save_path[:-3]+'__ae.h5')
        
    def load_saved_weights(self, weights_path=None):
        self._classification_model.load_weights(weights_path)
        self._autoencoder_model.load_weights(weights_path[:-3]+'__ae.h5')
