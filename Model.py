from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

def construct_model(num_classes):
    # load VGG16 model.
    input = Input(shape=(224, 224, 3))
    # include_top means remove the dense layers.
    resnet = ResNet50(include_top=False,
                     weights='imagenet',
                     input_tensor=input)

    # construt dense layers
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resnet.output_shape[1:]))
    top_model.add(Dense(num_classes))
    top_model.add(Activation('softmax'))

    # stack dense layers over VGG16 model.
    model = Model(input=resnet.input, output=top_model(resnet.output))

    # fix wights of ResNet
    for layer in model.layers[:len(model.layers) - 1]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    return model
