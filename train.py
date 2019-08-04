import Model
import Data

model = Model.construct_model(100)
(x_train, y_train), (x_test, y_test) = Data.get_datasets(224, 224)

print(x_train.shape, y_train.shape)

history = model.fit(
    x=x_train, y=y_train,
    validation_split=0.3,
    epochs=100
)
