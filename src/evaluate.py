import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, val_generator):
    loss, accuracy = model.evaluate(val_generator)
    print(f"Test accuracy: {accuracy:.2f}")
    return loss, accuracy

def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

def save_history(history, filename='training_history.csv'):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(filename, index=False)
