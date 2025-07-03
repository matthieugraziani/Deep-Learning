
import tensorflow as tf

def configure_gpu():
    """
    Configure l'utilisation optimale du GPU pour TensorFlow.
    Utilise la croissance mémoire dynamique pour éviter l'allocation complète.
    """
    
    # Vérification de la disponibilité GPU
    print("=== Configuration GPU ===")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    print(f"Version CUDA: {tf.test.is_built_with_cuda()}")
    
    # Configuration de la croissance mémoire GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Activation de la croissance mémoire pour chaque GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Utilisation de la stratégie par défaut pour éviter les conflits
            # Seulement utiliser MirroredStrategy si on a plusieurs GPUs
            if len(gpus) > 1:
                gpu_strategy = tf.distribute.MirroredStrategy()
                print(f"Utilisation de MirroredStrategy avec {gpu_strategy.num_replicas_in_sync} GPU(s)")
            else:
                gpu_strategy = tf.distribute.get_strategy()  # Stratégie par défaut
                print("Utilisation de la stratégie par défaut (1 GPU)")
            
            return gpu_strategy
            
        except RuntimeError as e:
            print(f"Erreur configuration GPU: {e}")
            return tf.distribute.get_strategy()
    else:
        print("Aucun GPU détecté, utilisation du CPU")
        return tf.distribute.get_strategy()

# Configuration
strategy = configure_gpu()

# Test de performance GPU vs CPU
def test_gpu_performance():
    """Test rapide de performance GPU vs CPU"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Opération matricielle simple pour tester
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        import time
        start = time.time()
        c = tf.matmul(a, b)
        tf.reduce_sum(c)  # Force l'exécution
        end = time.time()
        
        device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
        print(f"Temps d'exécution sur {device}: {end-start:.4f} secondes")

test_gpu_performance()