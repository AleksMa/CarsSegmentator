from keras import Input
from keras.models import Model
from keras.layers import BatchNormalization, concatenate, Conv2D, Dropout, MaxPooling2D, LeakyReLU, UpSampling2D


def conv(y, f):
    x = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(y)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def down(x):
    cs = []
    ds = []
    pool = x

    for f in [64, 128, 256, 512, 1024]:
        c = conv(pool, f)
        pool = MaxPooling2D()(c)
        d = Dropout(0.6)(c)
        cs += [c]
        ds += [d]

    return [ds[4], ds[3], cs[2], cs[1], cs[0]]


def up(x):
    up_c = Conv2D(
        512,
        2,
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D()(
            x[0]))

    up_c = LeakyReLU()(up_c)
    y = concatenate([x[1], up_c], axis=3)
    c = conv(y, 512)

    up_c = Conv2D(
        256,
        2,
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D()(c))

    up_c = BatchNormalization()(up_c)
    up_c = LeakyReLU()(up_c)
    up_c = Dropout(0.6)(up_c)
    y = concatenate([x[2], up_c], axis=3)
    c = conv(y, 256)

    up_c = Conv2D(
        128,
        2,
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D()(c))

    up_c = BatchNormalization()(up_c)
    up_c = LeakyReLU()(up_c)
    y = concatenate([x[3], up_c], axis=3)
    c = conv(y, 128)

    up_c = Conv2D(
        64,
        2,
        padding='same',
        kernel_initializer='he_normal')(
        UpSampling2D()(c))

    up_c = BatchNormalization()(up_c)
    up_c = LeakyReLU()(up_c)
    up_c = Dropout(0.6)(up_c)
    y = concatenate([x[4], up_c], axis=3)
    c = conv(y, 64)

    c = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(c)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)

    return c


def get_model(input_size):
    inputs = Input(input_size + (1,))
    x = down(inputs)
    c = up(x)
    outputs = Conv2D(5, 1, activation='softmax')(c)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
