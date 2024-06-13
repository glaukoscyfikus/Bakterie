import tensorflow as tf
from src.data_processing import create_generators
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_history, save_history

# Konfiguracja GPU dla TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Sprawdzenie dostępności GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Ścieżki do danych treningowych i walidacyjnych
train_dir = 'data/train'
val_dir = 'data/val'

# Ustawienia rozmiaru obrazów i batch size
image_size = (224, 224)
batch_size = 32
epochs = 50

# Tworzenie generatorów danych
train_generator, val_generator = create_generators(train_dir, val_dir, image_size, batch_size)

# Tworzenie modelu
model = create_model(image_size)

# Trenowanie modelu
history = train_model(model, train_generator, val_generator, epochs)

# Ewaluacja modelu
evaluate_model(model, val_generator)

# Zapisywanie historii treningu
save_history(history)

# Tworzenie wykresów historii treningu
plot_history(history)
