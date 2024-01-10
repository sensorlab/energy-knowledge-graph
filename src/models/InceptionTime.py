# CHECK INCEPTION TIME IMPLEMENTATION 
from NUK import F1Score
import tensorflow as tf
from tensorflow import keras


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, model_number, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500, lr=0.001):

        # define callbacks here
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.25, patience=7, min_lr=0.0001)
        early_stopping = keras.callbacks.EarlyStopping(patience=9, restore_best_weights=True, monitor='loss')

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = [reduce_lr, early_stopping]
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.model_number = model_number

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            self.verbose = verbose
            self.model.save_weights(self.output_directory / 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=['accuracy', F1Score])
        

        return model


    def fit(self, x_train, y_train, callback, validation):
       
       

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size
        
        callbacks2 = self.callbacks
        callbacks2.append(callback)
        self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=2, callbacks=callbacks2, validation_data=validation)


        self.model.save(self.output_directory / f'model_{self.model_number}.hdf5')

        keras.backend.clear_session()

        return None

    def predict(self, x_test):
        model_path = self.output_directory / f'model_{self.model_number}.hdf5'
        model = keras.models.load_model(model_path, custom_objects={'F1Score': F1Score})
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)

        return y_pred
