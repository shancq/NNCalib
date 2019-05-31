import tensorflow as tf

class CNNBlock(tf.keras.Model):
    def __init__(self, num_filters, k, data_format, strides=1, dilated_rate=1):
        super(CNNBlock, self).__init__()
        self.conv = tf.keras.layers.Conv1D(num_filters,
                                           k,
                                           strides=strides,
                                           padding="same",
                                           activation=tf.nn.leaky_relu,
                                           dilation_rate=dilated_rate,
                                           use_bias=False,
                                           data_format=data_format,
                                           kernel_initializer="he_normal", )

    def call(self, x, training=None, mask=None):
        return self.conv(x)


class DilateConvBlock(tf.keras.Model):
    def __init__(self, num_filters, k, data_format="channels_last"):
        super(DilateConvBlock, self).__init__()
        axis = -1 if data_format == "channels_last" else 1
        num_filters = num_filters//2
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
        self.conv21 = CNNBlock(num_filters, 1, data_format, strides=1, dilated_rate=1)
        self.conv31 = CNNBlock(num_filters, 1, data_format, strides=1, dilated_rate=1)
        self.conv41 = CNNBlock(num_filters, 1, data_format, strides=1, dilated_rate=1)
        self.conv51 = CNNBlock(num_filters, 1, data_format, strides=1, dilated_rate=1)
        self.conv61 = CNNBlock(num_filters, 1, data_format, strides=1, dilated_rate=1)
        self.conv2 = CNNBlock(num_filters, k, data_format, strides=1, dilated_rate=1)
        self.conv3 = CNNBlock(num_filters, k, data_format, strides=1, dilated_rate=2)
        self.conv4 = CNNBlock(num_filters, k, data_format, strides=1, dilated_rate=4)
        self.conv5 = CNNBlock(num_filters, k, data_format, strides=1, dilated_rate=8)
        self.conv6 = CNNBlock(num_filters, k, data_format, strides=1, dilated_rate=16)
        self.conv7 = CNNBlock(num_filters*2*5, 1, data_format, strides=1, dilated_rate=1)
        self.gp = tf.keras.layers.GlobalAveragePooling1D(data_format)
        self.fc1 = tf.keras.layers.Dense(num_filters * 2, activation="selu")
        self.fc2 = tf.keras.layers.Dense(num_filters * 5 * 2, activation="sigmoid")
        self.wg = tf.keras.layers.Conv1D(1, 1, strides=1, padding="same", activation=tf.nn.softmax,
                                         data_format=data_format, kernel_initializer="he_normal", )

    def call(self, x, training=True, mask=None):
        leng = tf.shape(x)[1]
        h = self.batchnorm1(x, training=training)
        t1 = self.conv2(self.conv21(h))
        t2 = self.conv3(self.conv31(h))
        t3 = self.conv4(self.conv41(h))
        t4 = self.conv5(self.conv51(h))
        t5 = self.conv6(self.conv61(h))
        output = self.conv7(tf.concat([t1, t2, t3, t4, t5], axis=-1))
        s = self.gp(output)
        e = self.fc2(self.fc1(s))
        output = output * tf.tile(tf.expand_dims(e, 1), [1, leng, 1])
        S = x * tf.tile(self.wg(output), [1, 1, tf.shape(x)[2]])
        return tf.concat([output, S], axis=-1)


class ConveCsNet(tf.keras.Model):
    def __init__(self, num_of_blocks, comp_dims, ori_dims, num_filters, k, data_format="channels_last"):
        super(ConveCsNet, self).__init__()
        # self.a = tf.get_variable('a', dtype=tf.float32, shape=[1], initializer=tf.random_uniform_initializer)
        # self.b = tf.get_variable('b', dtype=tf.float32, shape=[1], initializer=tf.random_uniform_initializer)
        self.scale = ori_dims // comp_dims
        self.num_of_blocks = num_of_blocks
        # encoder
        self.denser = tf.keras.layers.Dense(comp_dims // 2,
                                            use_bias=False,
                                            kernel_initializer="he_normal")
        self.densei = tf.keras.layers.Dense(comp_dims // 2,
                                            use_bias=False,
                                            kernel_initializer="he_normal")
        self.up = tf.keras.layers.UpSampling1D(self.scale)
        self.conv1 = CNNBlock(1, 1, data_format, strides=1, dilated_rate=1)
        self.convup = CNNBlock(num_filters * 5, 3, data_format, strides=1, dilated_rate=1)
        axis = -1 if data_format == "channels_last" else 1
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis)
        self.cs_blocks = []
        for i in range(num_of_blocks):
            self.cs_blocks.append(DilateConvBlock(num_filters * 5,
                                                  k,
                                                  data_format, ))

    def call(self, x, training=True, mask=None):
        # Linear encoding
        batch_size, ori_dims = tf.shape(x)
        xreal = tf.slice(x, [0, 0], [batch_size, ori_dims // 2])
        ximag = tf.slice(x, [0, ori_dims // 2], [batch_size, ori_dims // 2])

        yreal = self.denser(xreal) - self.densei(ximag)
        yimag = self.denser(ximag) + self.densei(xreal)

        enc = tf.concat([yreal, yimag], 1)
        # decoding
        y = self.up(tf.expand_dims(enc, -1))
        y = self.batchnorm1(self.convup(y), training=training)
        for i in range(self.num_of_blocks):
            y = self.cs_blocks[i](y, training=training)
        return self.conv1(y), enc
