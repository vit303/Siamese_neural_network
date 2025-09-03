import tensorflow as tf
from keras import models
import numpy as np
import random
import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt  # Добавлено для отображения изображений

# Устанавливаем использование GPU (упрощённая версия из основного кода)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Найдено GPU: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не обнаружены, используется CPU")

# Определение функции euclidean_distance (должно совпадать с вашим основным кодом)
# Убрал декоратор, чтобы избежать ошибок с версией Keras/TensorFlow
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Загрузка тестовых данных MNIST
def load_test_data():
    """Загрузка и подготовка тестовых данных MNIST"""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

# Функция для тестирования (адаптирована из вашего кода)
def test_model(model, x_test, y_test, num_tests=10):
    """Тестирование модели на нескольких примерах с разными и одинаковыми цифрами"""
    print("\nТестирование модели (смешанные пары)...")
    
    digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    
    for i in range(num_tests):
        # С вероятностью 0.5 выбираем пару одинаковых цифр
        if random.random() < 0.5:
            d = random.randint(0, 9)
            idx1 = random.choice(digit_indices[d])
            idx2 = random.choice(digit_indices[d])
            while idx1 == idx2:
                idx2 = random.choice(digit_indices[d])
            expected = "одинаковые"
        else:
            d1 = random.randint(0, 9)
            d2 = (d1 + random.randint(1, 9)) % 10
            idx1 = random.choice(digit_indices[d1])
            idx2 = random.choice(digit_indices[d2])
            expected = "разные"
        
        img1 = x_test[idx1]
        img2 = x_test[idx2]
        label1 = y_test[idx1]
        label2 = y_test[idx2]
        
        distance = model.predict([np.expand_dims(img1, 0), np.expand_dims(img2, 0)], verbose=0)[0][0]
        similarity = 1 - min(distance, 1.0)
        
        print(f"Тест {i+1}:")
        print(f"  Изображение 1: цифра {label1}")
        print(f"  Изображение 2: цифра {label2}")
        print(f"  Расстояние: {distance:.3f}")
        print(f"  Схожесть: {similarity:.3f}")
        print(f"  Ожидаемый результат: {expected}")
        
        # Отображение изображений с подписями
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img1.squeeze(), cmap='gray')
        plt.title(f"Цифра {label1}")
        plt.xticks([])  # Скрываем ticks по X
        plt.yticks([])  # Скрываем ticks по Y
        plt.xlabel(f"Расстояние: {distance:.3f}\nСхожесть: {similarity:.3f}", fontsize=10)
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2.squeeze(), cmap='gray')
        plt.title(f"Цифра {label2}")
        plt.xticks([])  # Скрываем ticks по X
        plt.yticks([])  # Скрываем ticks по Y
        plt.xlabel(f"Ожидаемый: {expected}", fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        print()

# Главная функция для тестирования
def main():
    """Запуск тестирования модели"""
    print("=== Тестирование сиамской сети на MNIST ===")
    
    try:
        # Загрузка модели
        model_path = 'siamese_mnist_model.h5'  # Укажите путь, если модель в другом месте
        if not os.path.exists(model_path):
            print(f"Модель не найдена по пути: {model_path}")
            return
        
        print(f"Загрузка модели из {model_path}...")
        # Добавлено custom_objects для загрузки модели с custom функцией
        model = models.load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance}, compile=False)
        
        # Загрузка тестовых данных
        print("Загрузка тестовых данных MNIST...")
        x_test, y_test = load_test_data()
        
        # Тестирование
        test_model(model, x_test, y_test, num_tests=10)  # Можно изменить num_tests
        
        print("Тестирование завершено!")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

# Запуск
if __name__ == "__main__":
    main()