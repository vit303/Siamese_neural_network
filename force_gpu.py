import os
import ctypes

def verify_cudnn_installation():
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    
    print("=== ПРОВЕРКА УСТАНОВКИ CUDNN ===")
    
    # Проверяем наличие файлов
    required_files = {
        'bin': ['cudnn64_8.dll'],
        'include': ['cudnn.h'],
        'lib/x64': ['cudnn.lib']
    }
    
    all_files_exist = True
    
    for folder, files in required_files.items():
        for file in files:
            file_path = os.path.join(cuda_path, folder, file)
            if os.path.exists(file_path):
                print(f"✅ {file} - найден в {folder}")
            else:
                print(f"❌ {file} - отсутствует в {folder}")
                all_files_exist = False
    
    # Проверяем загрузку DLL
    print("\n=== ПРОВЕРКА ЗАГРУЗКИ DLL ===")
    try:
        cudnn_path = os.path.join(cuda_path, "bin", "cudnn64_8.dll")
        cudnn_dll = ctypes.WinDLL(cudnn_path)
        print("✅ cudnn64_8.dll загружена успешно")
        
        # Получаем версию cuDNN
        try:
            cudnn_version = ctypes.c_int()
            cudnn_dll.cudnnGetVersion(ctypes.byref(cudnn_version))
            print(f"✅ Версия cuDNN: {cudnn_version.value}")
        except:
            print("ℹ️  Не удалось получить версию cuDNN")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки cudnn64_8.dll: {e}")
        all_files_exist = False
    
    return all_files_exist

if __name__ == "__main__":
    success = verify_cudnn_installation()
    
    if success:
        print("\n🎉 cuDNN успешно установлен и работает!")
        print("Перезапустите Python и проверьте TensorFlow:")
        print("import tensorflow as tf")
        print("print(tf.config.list_physical_devices('GPU'))")
    else:
        print("\n❌ cuDNN не установлен правильно")
        print("Скачайте вручную с: https://developer.nvidia.com/cudnn")