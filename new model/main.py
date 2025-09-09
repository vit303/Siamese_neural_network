import tensorflow as tf
from keras import layers, models
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Устанавливаем использование GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Разрешаем рост памяти GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Устанавливаем стратегию распределения
        strategy = tf.distribute.MirroredStrategy()
        print(f'Number of devices: {strategy.num_replicas_in_sync}')
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.get_strategy()
    print("GPU не обнаружены, используется CPU")

# 1. Функции для создания модели
def create_base_network(input_shape):
    """Создает базовую сеть-энкодер для дефектов поверхности"""
    with strategy.scope():
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu')  # Выходное embedding-пространство
        ])
    return model

def euclidean_distance(vectors):
    """Вычисляет евклидово расстояние между двумя векторов"""
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
    with strategy.scope():
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

# 2. Функции для подготовки данных NEU Surface Defect Dataset
def extract_defect_class(filename):
    """Извлекает класс дефекта из имени файла"""
    # Пример: crazing_1.jpg.rf.06382e752d1e00ca... -> crazing
    match = re.match(r'^([a-zA-Z]+)_', filename)
    if match:
        return match.group(1).lower()
    return "unknown"

def load_neu_surface_defect_data(data_path='model/train'):
    """Загрузка и подготовка данных NEU Surface Defect Dataset"""
    images = []
    labels = []
    
    # Проверяем существование пути
    if not os.path.exists(data_path):
        # Пробуем альтернативные пути
        alternative_paths = [
            'model/train',
            '../model/train',
            './model/train',
            'train',
            '../train'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"Найдена папка с данными: {data_path}")
                break
        else:
            # Если ни один путь не найден, покажем доступные папки
            current_dir = os.getcwd()
            print(f"Текущая директория: {current_dir}")
            print("Доступные папки:")
            for item in os.listdir(current_dir):
                if os.path.isdir(item):
                    print(f"  - {item}")
            raise ValueError(f"Папка {data_path} не существует!")
    
    print(f"Загрузка данных из: {data_path}")
    
    # Собираем все файлы изображений
    image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        raise ValueError("Не найдено изображений в указанной папке!")
    
    print(f"Найдено {len(image_files)} изображений")
    
    for img_file in image_files:
        img_path = os.path.join(data_path, img_file)
        
        # Извлекаем класс дефекта из имени файла
        defect_class = extract_defect_class(img_file)
        
        try:
            # Чтение и предобработка изображения
            img = cv2.imread(img_path)
            if img is None:
                print(f"Не удалось загрузить: {img_file}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))  # Размер для NEU dataset
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(defect_class)
            
        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
            continue
    
    print(f"Успешно загружено {len(images)} изображений")
    print(f"Найдены классы: {set(labels)}")
    
    return np.array(images), np.array(labels)

def create_defect_pairs(x, y, num_pairs_per_class=300):
    """Создание пар изображений дефектов для обучения"""
    unique_classes = np.unique(y)
    print(f"Создание пар для классов: {unique_classes}")
    
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
    
    pairs = []
    labels = []
    
    for cls in unique_classes:
        indices = class_indices[cls]
        print(f"Класс {cls}: {len(indices)} изображений")
        
        # Positive pairs - одинаковые классы дефектов
        if len(indices) >= 2:
            for i in range(min(num_pairs_per_class, len(indices) * 2)):
                try:
                    idx1, idx2 = random.sample(list(indices), 2)
                    pairs.append([x[idx1], x[idx2]])
                    labels.append(1)  # 1 - одинаковые дефекты
                except ValueError:
                    break
        
        # Negative pairs - разные классы дефектов
        other_classes = [c for c in unique_classes if c != cls]
        for i in range(num_pairs_per_class):
            if other_classes and indices.size > 0:
                other_cls = random.choice(other_classes)
                other_indices = class_indices[other_cls]
                if other_indices.size > 0:
                    idx1 = random.choice(indices)
                    idx2 = random.choice(other_indices)
                    pairs.append([x[idx1], x[idx2]])
                    labels.append(0)  # 0 - разные дефекты
    
    pairs = np.array(pairs)
    labels = np.array(labels)
    
    print(f"Создано {len(pairs)} пар: {np.sum(labels==1)} положительных, {np.sum(labels==0)} отрицательных")
    
    return pairs, labels

# 3. Обучение модели
def train_siamese_network():
    """Полный процесс обучения сиамской сети для дефектов поверхности"""
    # Загрузка данных
    print("Загрузка данных NEU Surface Defect Dataset...")
    x_data, y_data = load_neu_surface_defect_data('model/train')
    
    if len(x_data) == 0:
        raise ValueError("Не удалось загрузить данные. Проверьте путь к dataset.")
    
    # Разделение на train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    
    # Создание пар
    print("Создание обучающих пар...")
    # Автоматически подбираем количество пар в зависимости от размера данных
    pairs_per_class = min(200, len(x_train) // len(np.unique(y_train)) // 2)
    train_pairs, train_labels = create_defect_pairs(x_train, y_train, num_pairs_per_class=pairs_per_class)
    test_pairs, test_labels = create_defect_pairs(x_test, y_test, num_pairs_per_class=pairs_per_class // 2)
    
    # Разделение на два массива
    x_train_1 = train_pairs[:, 0]
    x_train_2 = train_pairs[:, 1]
    x_test_1 = test_pairs[:, 0]
    x_test_2 = test_pairs[:, 1]
    
    # Создание модели
    print("Создание сиамской сети...")
    input_shape = (200, 200, 3)  # Размер изображений NEU dataset
    
    with strategy.scope():
        siamese_net = create_siamese_network(input_shape)
        siamese_net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            loss=contrastive_loss, 
            metrics=['accuracy']
        )
    
    # Параметры обучения
    batch_size = 32 * strategy.num_replicas_in_sync
    epochs = 50  # Уменьшено для быстрого тестирования
    
    print(f"Используется batch size: {batch_size}")
    print(f"Количество реплик: {strategy.num_replicas_in_sync}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best_siamese_defect_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]
    
    # Обучение
    print("Начало обучения...")
    history = siamese_net.fit(
        [x_train_1, x_train_2], train_labels,
        validation_data=([x_test_1, x_test_2], test_labels),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )
    
    return siamese_net, history, (x_test, y_test)

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
    plt.savefig('training_history.png')
    plt.show()

# 5. Сохранение модели
def save_model(model, history):
    """Сохранение модели и истории обучения"""
    # Сохранение всей модели
    model.save('siamese_surface_defect_model.h5')
    print("Модель сохранена как 'siamese_surface_defect_model.h5'")
    
    # Сохранение истории обучения
    np.save('surface_defect_training_history.npy', history.history)
    print("История обучения сохранена как 'surface_defect_training_history.npy'")
    
    # Сохранение базовой сети
    base_network = model.layers[2]  # Базовая сеть
    base_network.save('surface_defect_base_network.h5')
    print("Базовая сеть сохранена как 'surface_defect_base_network.h5'")

# 6. Функция для тестирования
def test_model(model, x_test, y_test, num_tests=5):
    """Тестирование модели на нескольких примерах дефектов"""
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
        print(f"  Изображение 1: дефект {label1}")
        print(f"  Изображение 2: дефект {label2}")
        print(f"  Расстояние: {distance:.3f}")
        print(f"  Схожесть: {similarity:.3f}")
        print(f"  Ожидаемый результат: {'одинаковые' if label1 == label2 else 'разные'}")
        print()

# 7. Главная функция
def main():
    """Запуск полного процесса"""
    print("=== Обучение сиамской сети на NEU Surface Defect Dataset ===")
    print(f"Используется стратегия: {type(strategy).__name__}")
    print(f"Доступные GPU: {len(gpus) if gpus else 0}")
    
    # Покажем текущую структуру папок для отладки
    current_dir = os.getcwd()
    print(f"Текущая директория: {current_dir}")
    print("Содержимое текущей директории:")
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}")
        else:
            print(f"  📄 {item}")
    
    try:
        # Обучение модели
        model, history, (x_test, y_test) = train_siamese_network()
        
        # Визуализация результатов
        plot_training_history(history)
        
        # Сохранение модели
        save_model(model, history)
        
        # Тестирование
        test_model(model, x_test, y_test, num_tests=8)
        
        print("Обучение завершено успешно!")
        print("Модель готова для сравнения изображений дефектов поверхности!")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Попробуйте уменьшить размер батча или количество данных")
        import traceback
        traceback.print_exc()

# Запуск
if __name__ == "__main__":
    main()