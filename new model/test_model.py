import tensorflow as tf
from keras import models
import numpy as np
import random
import os
import cv2
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Устанавливаем использование GPU
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

# Определение функции euclidean_distance
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

# Функция для извлечения класса дефекта из имени файла
def extract_defect_class(filename):
    """Извлекает класс дефекта из имени файла"""
    match = re.match(r'^([a-zA-Z]+)_', filename)
    if match:
        return match.group(1).lower()
    return "unknown"

# Загрузка тестовых данных
def load_test_data(data_path='model/valid'):
    """Загрузка и подготовка тестовых данных"""
    images = []
    labels = []
    
    # Проверяем существование пути
    if not os.path.exists(data_path):
        # Пробуем альтернативные пути
        alternative_paths = [
            'model/valid',
            '../model/valid',
            './model/valid',
            'valid',
            '../valid',
            'model/train',  # Если нет valid, используем train
            'train'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"Найдена папка с данными: {data_path}")
                break
        else:
            raise ValueError(f"Папка с данными не найдена!")
    
    print(f"Загрузка тестовых данных из: {data_path}")
    
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
            img = cv2.resize(img, (200, 200))
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(defect_class)
            
        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
            continue
    
    print(f"Успешно загружено {len(images)} изображений")
    print(f"Найдены классы: {set(labels)}")
    
    return np.array(images), np.array(labels)

# Функция для тестирования модели
def test_model(model, x_test, y_test, num_tests=10):
    """Тестирование модели на нескольких примерах дефектов"""
    print("\nТестирование модели на дефектах поверхности...")
    
    # Получаем индексы по классам
    unique_classes = np.unique(y_test)
    class_indices = {cls: np.where(y_test == cls)[0] for cls in unique_classes}
    
    for i in range(num_tests):
        # С вероятностью 0.5 выбираем пару одинаковых дефектов
        if random.random() < 0.5:
            # Одинаковые дефекты
            cls = random.choice(unique_classes)
            indices = class_indices[cls]
            if len(indices) >= 2:
                idx1, idx2 = random.sample(list(indices), 2)
                expected = "одинаковые"
            else:
                continue
        else:
            # Разные дефекты
            cls1 = random.choice(unique_classes)
            cls2 = random.choice([c for c in unique_classes if c != cls1])
            if class_indices[cls1].size > 0 and class_indices[cls2].size > 0:
                idx1 = random.choice(class_indices[cls1])
                idx2 = random.choice(class_indices[cls2])
                expected = "разные"
            else:
                continue
        
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
        print(f"  Ожидаемый результат: {expected}")
        
        # Отображение изображений
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title(f"Дефект: {label1}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title(f"Дефект: {label2}")
        plt.axis('off')
        
        plt.suptitle(f"Расстояние: {distance:.3f} | Схожесть: {similarity:.3f}\nОжидаемый: {expected}", 
                    fontsize=12)
        plt.tight_layout()
        plt.show()
        
        print()

# Функция для тестирования конкретных пар изображений
def test_specific_pairs(model, image_paths):
    """Тестирование конкретных пар изображений"""
    print("\nТестирование конкретных пар изображений...")
    
    def load_and_preprocess_image(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 200))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, 0)
    
    for i, (img1_path, img2_path) in enumerate(image_paths):
        try:
            img1 = load_and_preprocess_image(img1_path)
            img2 = load_and_preprocess_image(img2_path)
            
            # Предсказание
            distance = model.predict([img1, img2], verbose=0)[0][0]
            similarity = 1 - min(distance, 1.0)
            
            print(f"Пара {i+1}:")
            print(f"  Изображение 1: {os.path.basename(img1_path)}")
            print(f"  Изображение 2: {os.path.basename(img2_path)}")
            print(f"  Расстояние: {distance:.3f}")
            print(f"  Схожесть: {similarity:.3f}")
            print(f"  Вероятность одинакового дефекта: {similarity*100:.1f}%")
            
            # Отображение изображений
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img1[0])
            plt.title(f"Изображение 1\n{os.path.basename(img1_path)}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img2[0])
            plt.title(f"Изображение 2\n{os.path.basename(img2_path)}")
            plt.axis('off')
            
            plt.suptitle(f"Расстояние: {distance:.3f} | Схожесть: {similarity:.3f}", 
                        fontsize=12)
            plt.tight_layout()
            plt.show()
            
            print()
            
        except Exception as e:
            print(f"Ошибка при обработке пары {i+1}: {e}")
            continue

# Главная функция для тестирования
def main():
    """Запуск тестирования модели"""
    print("=== Тестирование сиамской сети на NEU Surface Defect Dataset ===")
    
    try:
        # Загрузка модели
        model_path = 'siamese_surface_defect_model.h5'
        if not os.path.exists(model_path):
            print(f"Модель не найдена по пути: {model_path}")
            print("Попытка найти альтернативные пути...")
            
            alternative_paths = [
                'best_siamese_defect_model.h5',
                'surface_defect_base_network.h5',
                '../siamese_surface_defect_model.h5'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Найдена модель: {model_path}")
                    break
            else:
                print("Модель не найдена. Убедитесь, что она существует.")
                return
        
        print(f"Загрузка модели из {model_path}...")
        model = models.load_model(model_path, custom_objects={
            'euclidean_distance': euclidean_distance,
            'contrastive_loss': None  # Может потребоваться для совместимости
        }, compile=False)
        
        # Загрузка тестовых данных
        print("Загрузка тестовых данных...")
        x_test, y_test = load_test_data()
        
        # Тестирование на случайных парах
        test_model(model, x_test, y_test, num_tests=8)
        
        # Дополнительно: тестирование конкретных пар (раскомментируйте если нужно)
        # specific_pairs = [
        #     ('path/to/image1.jpg', 'path/to/image2.jpg'),
        #     ('path/to/image3.jpg', 'path/to/image4.jpg')
        # ]
        # test_specific_pairs(model, specific_pairs)
        
        print("Тестирование завершено!")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

# Запуск
if __name__ == "__main__":
    main()