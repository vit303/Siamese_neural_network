import tensorflow as tf
from keras import layers, models
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Функции для создания модели
def create_base_network(input_shape):
    """Создает базовую сеть-энкодер"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu')  # Выходное embedding-пространство
    ])
    return model

def euclidean_distance(vectors):
    """Вычисляет евклидово расстояние между двумя векторами"""
    vector1, vector2 = vectors
    sum_square = tf.reduce_sum(tf.square(vector1 - vector2), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    """Контрастная функция потерь"""
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_siamese_network(input_shape):
    """Создает полную сиамскую сеть"""
    # Два входных изображения
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Базовая сеть (общие веса)
    base_network = create_base_network(input_shape)
    
    # Выходы для двух изображений
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Вычисление расстояния между embedding'ами
    distance = layers.Lambda(euclidean_distance, output_shape=(1,))([processed_a, processed_b])
    
    # Создание модели
    model = models.Model([input_a, input_b], distance)
    return model

# 2. Функции для подготовки данных
def load_and_preprocess_data():
    """Загрузка и подготовка данных MNIST"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Нормализация и добавление размерности канала
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return (x_train, y_train), (x_test, y_test)

def create_pairs(x, y, num_pairs_per_class=1000):
    """Создание пар изображений для обучения"""
    num_classes = len(np.unique(y))
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    pairs = []
    labels = []
    
    for d in range(num_classes):
        # Positive pairs - одинаковые цифры
        for i in range(num_pairs_per_class):
            idx1 = random.choice(digit_indices[d])
            idx2 = random.choice(digit_indices[d])
            while idx1 == idx2:  # Убедимся, что это разные изображения
                idx2 = random.choice(digit_indices[d])
            pairs.append([x[idx1], x[idx2]])
            labels.append(1)  # 1 - одинаковые
        
        # Negative pairs - разные цифры
        for i in range(num_pairs_per_class):
            d1 = d
            d2 = (d + random.randint(1, num_classes-1)) % num_classes
            idx1 = random.choice(digit_indices[d1])
            idx2 = random.choice(digit_indices[d2])
            pairs.append([x[idx1], x[idx2]])
            labels.append(0)  # 0 - разные
    
    return np.array(pairs), np.array(labels)

# 3. Обучение модели
def train_siamese_network():
    """Полный процесс обучения сиамской сети"""
    # Загрузка данных
    print("Загрузка данных MNIST...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Уменьшим количество пар для быстрого тестирования
    print("Создание обучающих пар...")
    train_pairs, train_labels = create_pairs(x_train, y_train, num_pairs_per_class=500)
    test_pairs, test_labels = create_pairs(x_test, y_test, num_pairs_per_class=100)
    
    # Разделение на два массива
    x_train_1 = train_pairs[:, 0]
    x_train_2 = train_pairs[:, 1]
    x_test_1 = test_pairs[:, 0]
    x_test_2 = test_pairs[:, 1]
    
    # Создание модели
    print("Создание сиамской сети...")
    input_shape = (28, 28, 1)
    siamese_net = create_siamese_network(input_shape)
    siamese_net.compile(
        optimizer='adam', 
        loss=contrastive_loss, 
        metrics=['accuracy']
    )
    
    # Параметры обучения
    batch_size = 64
    epochs = 150  # Уменьшим для тестирования
    
    # Обучение
    print("Начало обучения...")
    history = siamese_net.fit(
        [x_train_1, x_train_2], train_labels,
        validation_data=([x_test_1, x_test_2], test_labels),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    
    return siamese_net, history

# 4. Визуализация результатов
def plot_training_history(history):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 5. Сохранение модели
def save_model(model, history):
    """Сохранение модели и истории обучения"""
    # Сохранение всей модели
    model.save('siamese_mnist_model.h5')
    print("Модель сохранена как 'siamese_mnist_model.h5'")
    
    # Сохранение истории обучения
    np.save('training_history.npy', history.history)
    print("История обучения сохранена как 'training_history.npy'")
    
    # Сохранение базовой сети
    base_network = model.layers[2]  # Базовая сеть
    base_network.save('base_network_model.h5')
    print("Базовая сеть сохранена как 'base_network_model.h5'")

# 6. Функция для тестирования
def test_model(model, x_test, y_test, num_tests=5):
    """Тестирование модели на нескольких примерах"""
    print("\nТестирование модели...")
    
    for i in range(num_tests):
        # Выбираем случайные изображения
        idx1 = random.randint(0, len(x_test) - 1)
        idx2 = random.randint(0, len(x_test) - 1)
        
        img1 = x_test[idx1]
        img2 = x_test[idx2]
        label1 = y_test[idx1]
        label2 = y_test[idx2]
        
        # Предсказание
        distance = model.predict([np.expand_dims(img1, 0), np.expand_dims(img2, 0)], verbose=0)[0][0]
        similarity = 1 - min(distance, 1.0)
        
        print(f"Тест {i+1}:")
        print(f"  Изображение 1: цифра {label1}")
        print(f"  Изображение 2: цифра {label2}")
        print(f"  Расстояние: {distance:.3f}")
        print(f"  Схожесть: {similarity:.3f}")
        print(f"  Ожидаемый результат: {'одинаковые' if label1 == label2 else 'разные'}")
        print()

# 7. Главная функция
def main():
    """Запуск полного процесса"""
    print("=== Обучение сиамской сети на MNIST ===")
    
    try:
        # Обучение модели
        model, history = train_siamese_network()
        
        # Визуализация результатов
        plot_training_history(history)
        
        # Сохранение модели
        save_model(model, history)
        
        # Загрузка тестовых данных для демонстрации
        (_, _), (x_test, y_test) = load_and_preprocess_data()
        
        # Тестирование
        test_model(model, x_test, y_test)
        
        print("Обучение завершено успешно!")
        print("Модель готова для сравнения любых рукописных цифр!")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Попробуйте уменьшить размер батча или количество данных")

# Запуск
if __name__ == "__main__":
    main()