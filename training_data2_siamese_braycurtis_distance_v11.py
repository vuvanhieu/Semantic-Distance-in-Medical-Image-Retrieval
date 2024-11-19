import os
import csv
import numpy as np
import tensorflow as tf
import time
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras.mixed_precision import set_global_policy, LossScaleOptimizer
from tensorflow.keras.mixed_precision import experimental as mixed_precision  # For mixed precision training
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, silhouette_score  # Import bổ sung
from sklearn.metrics import pairwise_distances_argmin_min
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
import matplotlib.cm as cm
from tensorflow.keras.regularizers import l2
import gc
import tensorflow.keras.backend as K
import random
from collections import Counter
from tensorflow.keras.mixed_precision import set_global_policy, Policy

# Set CUDA allocator to avoid memory fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable XLA for devices
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Ensure memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Enable XLA compilation for better performance
try:
    tf.config.optimizer.set_jit(True)  # Enable XLA (Accelerated Linear Algebra)
    print("XLA JIT compilation enabled.")
except Exception as e:
    print(f"Failed to enable XLA: {e}")

# Disable mixed precision for stability, or enable it based on need
use_mixed_precision = True  # Change this to False if mixed precision isn't required

if use_mixed_precision:
    policy = Policy('mixed_float16')  # Enable mixed precision to reduce memory usage
    set_global_policy(policy)
    print("Mixed precision training enabled with float16.")
else:
    set_global_policy('float32')  # Use float32 for stability
    print("Mixed precision disabled, using float32.")

# Check GPU availability
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())

# Function to test TensorFlow with JIT
@tf.function(jit_compile=True)  # Enable JIT for this function
def sample_computation(x):
    return tf.reduce_sum(x ** 2)

# Run a sample computation to ensure everything works
x = tf.random.normal([1000, 1000])
result = sample_computation(x)
print(f"Sample computation result: {result.numpy()}")

def clear_memory():
    K.clear_session()
    gc.collect()
                  
# Sửa từ điển DISTANCE_METRICS để truyền đúng kmeans và siamese_model
def create_distance_metrics(results_dir):
    """
    Create a dictionary of distance metrics functions.
    """
    def compute_distance_with_full_data(query, metric):
        X_train = np.load(os.path.join(results_dir, 'X_train.npy'))
        print(f"Calculating {metric} distance with the entire dataset...")
        return cdist(query.reshape(1, -1), X_train, metric=metric).flatten()
    
    return {
        "euclidean": lambda query: compute_distance_with_full_data(query, 'euclidean'),
        "manhattan": lambda query: compute_distance_with_full_data(query, 'cityblock'),
        "cosine": lambda query: compute_distance_with_full_data(query, 'cosine'),
        "chebyshev": lambda query: compute_distance_with_full_data(query, 'chebyshev'),
        "braycurtis": lambda query: compute_distance_with_full_data(query, 'braycurtis'),
        "siamese": lambda query: siamese_distance(query, results_dir)
    }


def siamese_distance(query_image, results_dir):
    """
    Calculate the distance between the query_image and all samples in the training set using a pre-trained Siamese model.
    """
    # Load the entire training dataset
    X_train = np.load(os.path.join(results_dir, 'X_train.npy'))

    # Load the pre-trained Siamese model weights
    weights_path = os.path.join(results_dir, 'best_weights.h5')
    siamese_model = build_siamese_network(query_image.shape, results_dir)
    siamese_model.load_weights(weights_path)

    # Prepare the query and training data for the Siamese network
    query_image_expanded = np.expand_dims(query_image, axis=1)  # Add dimension for compatibility
    X_train_expanded = np.expand_dims(X_train, axis=1)  # Add dimension for compatibility

    # Predict distances using the Siamese model
    distances = siamese_model.predict([np.tile(query_image_expanded, (len(X_train), 1, 1)), X_train_expanded])

    # Normalize distances
    min_dist, max_dist = np.min(distances), np.max(distances)
    normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-10)

    return normalized_distances.flatten()


def braycurtis_distance(vects):
    """Calculate Bray-Curtis distance between two vectors."""
    x, y = vects
    num = K.sum(K.abs(x - y), axis=-1, keepdims=True)
    denom = K.sum(K.abs(x + y), axis=-1, keepdims=True)
    return num / (denom + K.epsilon())

   
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# Build Siamese network
def cosine_distance(vects):
    """Tính toán khoảng cách cosine giữa hai vector."""
    x, y = vects
    dot_product = K.sum(x * y, axis=-1, keepdims=True)
    norm_x = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))
    norm_y = K.sqrt(K.sum(K.square(y), axis=-1, keepdims=True))
    return 1 - dot_product / (norm_x * norm_y + K.epsilon())

# Function to build a deeper Siamese Network
def build_siamese_network(input_shape, results_dir):
    """
    Xây dựng mô hình mạng Siamese với các tầng Dense, BatchNormalization và Dropout.
    """
    # Khởi tạo mô hình CNN
    cnn = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4))
    ])

    # Định nghĩa đầu vào cho cặp ảnh
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    # Truyền dữ liệu qua CNN cho từng đầu vào
    processed_a = cnn(input_a)
    processed_b = cnn(input_b)

    # Tính khoảng cách Euclidean giữa hai đầu ra
    distance = tf.keras.layers.Lambda(braycurtis_distance)([processed_a, processed_b])
    # Khởi tạo mô hình Siamese với đầu vào kép
    model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=distance)
    # Khởi tạo optimizer với mixed precision và Loss Scaling
    base_optimizer = Adam(learning_rate=1e-5)
    optimizer = LossScaleOptimizer(base_optimizer, dynamic=True)
    # Biên dịch mô hình với optimizer và loss thích hợp
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save model plot only once
    plot_path = os.path.join(results_dir, "siamese_model_architecture.png")
    if not os.path.exists(plot_path):
        print(f"Saving model plot to {plot_path}")
        plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
    # else:
    #     print(f"Model plot already exists at {plot_path}")
    
    return model


def plot_precision_at_k(k_values, precision_values_by_metric, results_dir):
    num_metrics = len(precision_values_by_metric)
    bar_width = 0.8 / max(1, num_metrics)
    k_values_range = np.arange(len(k_values))

    plt.figure(figsize=(18, 8))
    colors = cm.get_cmap('tab20', num_metrics)

    with open(os.path.join(results_dir, 'precision_at_k_values.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'K', 'Precision'])

        for idx, (metric_name, values) in enumerate(precision_values_by_metric.items()):
            if len(values) != len(k_values):
                print(f"Warning: Mismatch in values for {metric_name}. Skipping...")
                continue

            bars = plt.bar(
                k_values_range + idx * bar_width, values,
                width=bar_width, label=metric_name, color=colors(idx)
            )

            # Thêm chỉ số trên đỉnh mỗi cột và ghi vào tệp CSV
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                         ha='center', va='bottom', rotation=90)
                writer.writerow([metric_name, k_values[i], height])

    plt.xticks(k_values_range + (num_metrics - 1) * bar_width / 2, k_values, rotation=45)
    # plt.xlabel('Top-K', rotation=90, labelpad=15)
    plt.xlabel('Top-K', labelpad=15)
    plt.ylabel('Precision', rotation=90, labelpad=15)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_save_path = os.path.join(results_dir, 'Precision_at_k.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"Precision@K plot saved at {plot_save_path}")
    plt.close()

def plot_map_at_k(k_values, map_values_by_metric, results_dir):
    # Chuẩn hóa dữ liệu để loại bỏ NaN
    for metric_name, values in map_values_by_metric.items():
        map_values_by_metric[metric_name] = np.nan_to_num(values, nan=0.0)

    valid_metrics = {
        metric: values for metric, values in map_values_by_metric.items()
        if len(values) == len(k_values)
    }

    if not valid_metrics:
        print("Không có metric hợp lệ để vẽ biểu đồ mAP@K.")
        return

    plt.figure(figsize=(18, 8))
    bar_width = 0.8 / len(valid_metrics)
    k_range = np.arange(len(k_values))

    with open(os.path.join(results_dir, 'map_at_k_values.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'K', 'mAP'])

        for idx, (metric_name, values) in enumerate(valid_metrics.items()):
            bars = plt.bar(k_range + idx * bar_width, values, width=bar_width, label=metric_name)

            # Thêm chỉ số trên đỉnh mỗi cột và ghi vào tệp CSV
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                         ha='center', va='bottom', rotation=90)
                writer.writerow([metric_name, k_values[i], height])

    plt.xticks(k_range + bar_width / 2, k_values, rotation=45)
    # plt.xlabel('Top-K', rotation=90, labelpad=15)
    plt.xlabel('Top-K', labelpad=15)
    plt.ylabel('mAP', rotation=90, labelpad=15)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)

    plot_path = os.path.join(results_dir, 'mAP_at_k.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Lưu biểu đồ mAP@K tại: {plot_path}")
    plt.close()

def plot_f1_score_at_k(k_values, f1_values_by_metric, results_dir):
    """
    Plot F1-Score@K for different metrics.
    """
    num_metrics = len(f1_values_by_metric)
    bar_width = 0.8 / max(1, num_metrics)
    k_values_range = np.arange(len(k_values))

    plt.figure(figsize=(18, 8))
    colors = cm.get_cmap('tab20', num_metrics)

    with open(os.path.join(results_dir, 'f1_score_at_k_values.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'K', 'F1@K'])

        for idx, (metric_name, values) in enumerate(f1_values_by_metric.items()):
            if len(values) != len(k_values):
                print(f"Warning: Mismatch in values for {metric_name}. Skipping...")
                continue

            bars = plt.bar(
                k_values_range + idx * bar_width, values,
                width=bar_width, label=metric_name, color=colors(idx)
            )

            # Add values on top of the bars and write to CSV
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                         ha='center', va='bottom', rotation=90)
                writer.writerow([metric_name, k_values[i], height])

    plt.xticks(k_values_range + (num_metrics - 1) * bar_width / 2, k_values, rotation=45)
    plt.xlabel('Top-K', labelpad=15)
    plt.ylabel('F1-Score', rotation=90, labelpad=15)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_save_path = os.path.join(results_dir, 'F1_Score_at_k.png')
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"F1-Score@K plot saved at {plot_save_path}")
    plt.close()
    
    
def calculate_metrics_at_k(true_labels, predicted_labels, k):
    # Lấy top-K dự đoán cho đúng số lượng K yêu cầu
    predicted_labels = predicted_labels[:k]

    # Kiểm tra nếu nhãn dự đoán không chứa nhãn đúng
    if len(true_labels) != 1:
        raise ValueError("Each query should have exactly one true label.")

    # Kiểm tra xem nhãn đúng có nằm trong top-K dự đoán không
    match = int(true_labels[0] in predicted_labels)

    # Tính precision, recall và f1 thủ công
    precision = match / k
    recall = match  # Vì chỉ có một nhãn đúng
    f1 = 2 * precision * recall / (precision + recall + 1e-10)  # Tránh chia cho 0

    # Tính mAP@K
    try:
        map_k = average_precision_score([true_labels[0]], [predicted_labels])
    except ValueError as e:
        print(f"Error in calculating mAP: {e}")
        map_k = 0.0

    return precision, recall, f1, map_k

def calculate_ap_at_k(true_label, predicted_labels, k):
    """
    Tính AP cho một truy vấn với top-K dự đoán.
    """
    predicted_labels = predicted_labels[:k]
    num_correct = 0
    precision_sum = 0.0

    for i, label in enumerate(predicted_labels):
        if label == true_label:
            num_correct += 1
            precision_at_i = num_correct / (i + 1)
            precision_sum += precision_at_i

    if num_correct == 0:
        return 0.0  # Không có nhãn đúng nào trong top-K

    return precision_sum / num_correct

def calculate_map_at_k(true_labels, all_predicted_labels, k):
    """
    Tính mAP cho tất cả các truy vấn.
    """
    ap_sum = 0.0
    num_queries = len(true_labels)

    for true_label, predicted_labels in zip(true_labels, all_predicted_labels):
        ap = calculate_ap_at_k(true_label, predicted_labels, k)
        ap_sum += ap

    return ap_sum / num_queries if num_queries > 0 else 0.0


# def generate_pairs(samples, labels):
#     """
#     Tạo các cặp mẫu và nhãn cho mạng Siamese.
#     """
#     pairs = []
#     pair_labels = []

#     # Tạo các cặp cùng nhãn (positive pairs)
#     for i in range(len(samples)):
#         for j in range(i + 1, len(samples)):
#             if labels[i] == labels[j]:
#                 pairs.append([samples[i], samples[j]])
#                 pair_labels.append(1)

#     # Tạo các cặp khác nhãn (negative pairs)
#     for i in range(len(samples)):
#         for j in range(len(samples)):
#             if labels[i] != labels[j]:
#                 pairs.append([samples[i], samples[j]])
#                 pair_labels.append(0)

#     return np.array(pairs), np.array(pair_labels)

import numpy as np

def generate_pairs(samples, labels, max_positive_pairs=1000, max_negative_pairs=1000):
    """
    Generate pairs for a Siamese Network with memory-efficient sampling.
    Limits the number of positive and negative pairs to avoid memory issues.
    """
    pairs = []
    pair_labels = []

    # Create a mapping of labels to indices
    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    unique_labels = list(label_to_indices.keys())
    
    # Generate positive pairs
    for label, indices in label_to_indices.items():
        # Randomly sample positive pairs for the current label
        positive_pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positive_pairs.append((indices[i], indices[j]))
        
        # Shuffle and limit the number of positive pairs per label
        np.random.shuffle(positive_pairs)
        positive_pairs = positive_pairs[:max_positive_pairs]
        
        # Append positive pairs to the pairs list
        for (idx1, idx2) in positive_pairs:
            pairs.append([samples[idx1], samples[idx2]])
            pair_labels.append(1)

    # Generate negative pairs by pairing between different labels
    negative_pairs = []
    for i, label_a in enumerate(unique_labels):
        for label_b in unique_labels[i + 1:]:
            # Get indices for each label
            indices_a = label_to_indices[label_a]
            indices_b = label_to_indices[label_b]
            
            # Randomly sample pairs across classes
            negative_pairs.extend([(a, b) for a in indices_a for b in indices_b])
    
    # Shuffle and limit the number of negative pairs
    np.random.shuffle(negative_pairs)
    negative_pairs = negative_pairs[:max_negative_pairs]
    
    # Append negative pairs to the pairs list
    for (idx1, idx2) in negative_pairs:
        pairs.append([samples[idx1], samples[idx2]])
        pair_labels.append(0)
    
    return np.array(pairs), np.array(pair_labels)


def plot_history(history, model_name, results_dir):
    """
    Plot loss and accuracy from training history for a single model.
    This function now handles missing validation data gracefully.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    else:
        print(f"Warning: 'val_loss' not found for {model_name}. Skipping validation loss plot.")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy if available
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        else:
            print(f"Warning: 'val_accuracy' not found for {model_name}. Skipping validation accuracy plot.")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print(f"Warning: 'accuracy' not found for {model_name}. Skipping accuracy plot.")

    # Save the plot to the results directory with the model name in the filename
    plot_path = os.path.join(results_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path)
    print(f"Training history for {model_name} saved at {plot_path}")
    plt.close()


    
def siamese_data_generator(pairs, labels, batch_size):
    """Generator tạo batch dữ liệu cho Siamese Network."""
    while True:
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            yield ([batch_pairs[:, 0], batch_pairs[:, 1]], batch_labels)


def train_and_save_siamese(input_shape, X_train, y_train, X_val, y_val, results_dir):
    """
    Train a single Siamese model on the entire dataset without clustering.
    Save training parameters and plot training history.
    """
    batch_size = 64
    epochs = 200  # Maximum number of epochs

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Initialize CSV file for saving training information
    csv_path = os.path.join(results_dir, 'training_params.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Batch Size', 'Total Epochs', 'Stopped Epoch', 
            'Steps per Epoch', 'Validation Steps', 'Optimizer', 
            'Loss Function', 'Training Time (seconds)'
        ])

    print("Preparing training and validation pairs for the Siamese model...")

    # Generate pairs and labels for training and validation
    # train_pairs, train_labels = generate_pairs(X_train, y_train)
    # val_pairs, val_labels = generate_pairs(X_val, y_val)

    # Call generate_pairs for training and validation data
    train_pairs, train_labels = generate_pairs(X_train, y_train, max_positive_pairs=10000, max_negative_pairs=10000)
    val_pairs, val_labels = generate_pairs(X_val, y_val, max_positive_pairs=5000, max_negative_pairs=5000)


    # Create data generators for Siamese network
    train_generator = siamese_data_generator(train_pairs, train_labels, batch_size)
    val_generator = siamese_data_generator(val_pairs, val_labels, batch_size)

    steps_per_epoch = len(train_pairs) // batch_size
    validation_steps = len(val_pairs) // batch_size

    # Build the Siamese network
    siamese_model = build_siamese_network(input_shape, results_dir)

    # Set up callbacks for model saving and early stopping
    checkpoint_path = os.path.join(results_dir, 'best_weights.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start_time = time.time()

    # Train the Siamese model
    print("Training Siamese model on the entire dataset...")
    history = siamese_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )

    training_time = time.time() - start_time
    stopped_epoch = len(history.history['loss'])

    # Save training information to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            batch_size, epochs, stopped_epoch, steps_per_epoch, validation_steps,
            str(siamese_model.optimizer), siamese_model.loss, f"{training_time:.2f}"
        ])

    # Plot and save training history
    plot_history(history, "siamese_model", results_dir)
    print("Training completed. Model and history saved.")

    # Save model summary to text file
    model_summary_path = os.path.join(results_dir, 'model_summary.txt')
    with open(model_summary_path, 'w') as f:
        siamese_model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("Model summary saved.")

    # Clear memory
    clear_memory()

        
def calculate_p_at_k(true_label, predicted_labels, k):
    """
    Tính Precision cho một truy vấn với top-K dự đoán.
    """
    # Lọc các nhãn dự đoán trong top-K và chuyển thành danh sách
    predicted_labels_top_k = predicted_labels[:k].tolist()

    # Kiểm tra số lượng nhãn đúng có trong top-K dự đoán
    correct_predictions = predicted_labels_top_k.count(true_label)

    # Tính precision cho top-K
    return correct_predictions / k


def calculate_precision_at_k(true_labels, all_predicted_labels, k):
    """
    Tính Precision@K cho tất cả các truy vấn.
    """
    precision_sum = 0.0

    # Tính Precision cho từng truy vấn và cộng dồn
    for true_label, predicted_labels in zip(true_labels, all_predicted_labels):
        precision_sum += calculate_p_at_k(true_label, predicted_labels, k)

    # Trả về trung bình Precision@K
    return precision_sum / len(true_labels) if len(true_labels) > 0 else 0.0

# Additional function to calculate F1@K
def calculate_f1_at_k(true_labels, predicted_labels, k):
    """
    Calculate F1-Score@K for all queries.
    """
    precision = calculate_precision_at_k(true_labels, predicted_labels, k)
    recall = 1.0  # Each query has exactly one true label
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def apply_smote_tomek(X, y):
    """
    Apply SMOTE-Tomek for handling class imbalance.
    """
    if len(np.unique(y)) > 1:  # Check if there are multiple classes
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        print(f"Data after SMOTE-Tomek: {X_resampled.shape}, {y_resampled.shape}")
        return X_resampled, y_resampled
    else:
        print("Skipping SMOTE-Tomek: Only one class found.")
        return X, y  # Return the original data without resampling


def load_features_from_dir(directory, class_label):
    features = []
    labels = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return features, labels

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filepath.endswith('.npy'):
            feature = np.load(filepath)
            features.append(feature)
            labels.append(class_label)
    return features, labels

# Hàm để tải các đặc trưng từ các thư mục cho tất cả các tập dữ liệu
def load_all_data(base_dir, classes, encoder_path):
    X_train, y_train = [], []
    X_test, y_test = [], []
    X_val, y_val = [], []

    # Tải dữ liệu cho từng lớp và từng tập dữ liệu (train, test, val)
    for cls in classes:
        # Tập huấn luyện (train)
        train_dir = os.path.join(base_dir, 'train_Features', 'VGG16_features', cls)
        X_train_cls, y_train_cls = load_features_from_dir(train_dir, cls)
        X_train.extend(X_train_cls)
        y_train.extend(y_train_cls)

        # Tập kiểm tra (test)
        test_dir = os.path.join(base_dir, 'test_Features', 'VGG16_features', cls)
        X_test_cls, y_test_cls = load_features_from_dir(test_dir, cls)
        X_test.extend(X_test_cls)
        y_test.extend(y_test_cls)

        # Tập xác thực (val)
        val_dir = os.path.join(base_dir, 'val_Features', 'VGG16_features', cls)
        X_val_cls, y_val_cls = load_features_from_dir(val_dir, cls)
        X_val.extend(X_val_cls)
        y_val.extend(y_val_cls)

    # Chuyển đổi dữ liệu thành numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Áp dụng SMOTE-Tomek nếu được yêu cầu
    print("Applying SMOTE-Tomek to handle class imbalance...")
    X_train, y_train = apply_smote_tomek(X_train, y_train)
        
    # Mã hóa nhãn bằng LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    # Lưu encoder để tái sử dụng trong tương lai
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved at {encoder_path}")

    return X_train, y_train_encoded, X_test, y_test_encoded, X_val, y_val_encoded

def main():
    # Define directories and paths
    base_dir = '/data2/cmdir/home/hieuvv/IMAGES/data2/data2_SPLIT_FEATURE'
    results_dir = '/data2/cmdir/home/hieuvv/IMAGES/data2/training_data2_siamese_braycurtis_distance_v11'
    classes = ['VASC', 'DF', 'BKL', 'AKIEC', 'BCC', 'NV', 'MEL']  # Ensure correct class names
    encoder_path = os.path.join(results_dir, 'label_encoder.pkl')

    # Data paths
    x_train_path = os.path.join(results_dir, 'X_train.npy')
    y_train_path = os.path.join(results_dir, 'y_train.npy')
    x_test_path = os.path.join(results_dir, 'X_test.npy')
    y_test_path = os.path.join(results_dir, 'y_test.npy')
    x_val_path = os.path.join(results_dir, 'X_val.npy')
    y_val_path = os.path.join(results_dir, 'y_val.npy')

    # Create results directory if it does not exist
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load or preprocess the data
    if all(os.path.exists(path) for path in [x_train_path, y_train_path, x_test_path, y_test_path, x_val_path, y_val_path]):
        print("Loading preprocessed data from disk...")
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        X_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
    else:
        print("Step 1: Loading and preprocessing the data...")
        X_train, y_train, X_test, y_test, X_val, y_val = load_all_data(base_dir, classes, encoder_path)

        print(f"Loaded X_train with shape: {X_train.shape}")
        print(f"Loaded y_train with shape: {y_train.shape}")

        # Check for NaN or infinite values in data
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            raise ValueError("X_train contains NaN or infinite values!")
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            raise ValueError("y_train contains NaN or infinite values!")

        # Save processed data
        np.save(x_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)
        np.save(x_val_path, X_val)
        np.save(y_val_path, y_val)

    # Display data shapes and label distribution
    print(f"Data shapes: Train={X_train.shape}, Test={X_test.shape}, Val={X_val.shape}")
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Label distribution in y_train: {np.bincount(y_train)}")

    # Step 2: Train the Siamese model if needed
    model_weights_path = os.path.join(results_dir, 'best_weights.h5')
    
    if not os.path.exists(model_weights_path):
        print("Step 2: Training Siamese model on the entire dataset...")
        input_shape = (X_train.shape[1],)
        train_and_save_siamese(input_shape, X_train, y_train, X_val, y_val, results_dir)
    else:
        print("Model already trained and saved.")
        
    print("Step 5: Reloading test set...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    print("Step 6: Calculating distances using Siamese models...")
    k_values = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180]
    # k_values = [10]
    
    distance_metrics = create_distance_metrics(results_dir)

    precision_values_by_metric = {}
    map_values_by_metric = {}
    f1_values_by_metric = {}
    
    # Tính khoảng cách và đánh giá
    for metric_name in distance_metrics:
        precision_values_by_metric[metric_name] = []
        map_values_by_metric[metric_name] = []
        f1_values_by_metric[metric_name] = []
        all_predicted_labels = []
        true_labels = []

        for i, query_image in enumerate(X_test):
            query_image = query_image.reshape(1, -1)
            query_label = y_test[i]

            distances = distance_metrics[metric_name](query_image)
            nearest_indices = np.argsort(distances)
            print(f"Size of nearest_indices: {len(nearest_indices)}, Size of y_train_encoded: {len(y_train)}")

            valid_indices = nearest_indices[nearest_indices < len(y_train)]
            top_k_labels = y_train[valid_indices]

            all_predicted_labels.append(top_k_labels)
            true_labels.append(query_label)

        for k in k_values:
            top_k_predictions = [pred[:k] for pred in all_predicted_labels]
            precision_at_k = calculate_precision_at_k(true_labels, top_k_predictions, k)
            map_at_k = calculate_map_at_k(true_labels, top_k_predictions, k)
            f1_at_k = calculate_f1_at_k(true_labels, top_k_predictions, k)
            
            precision_values_by_metric[metric_name].append(precision_at_k)
            map_values_by_metric[metric_name].append(map_at_k)
            f1_values_by_metric[metric_name].append(f1_at_k)
            
            # print(f"Metric: {metric_name}, K={k}, Precision@K: {precision_at_k:.4f}, mAP@K: {map_at_k:.4f}")
            print(f"Metric: {metric_name}, K={k}, Precision@K: {precision_at_k:.4f}, "
                  f"mAP@K: {map_at_k:.4f}, F1@K: {f1_at_k:.4f}")
            
    query_results_csv_path = os.path.join(results_dir, 'query_results.csv')
    with open(query_results_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['Metric', 'K', 'Precision@K', 'mAP@K'])
        writer.writerow(['Metric', 'K', 'Precision@K', 'mAP@K', 'F1@K'])
        for metric_name in distance_metrics:
            for i, k in enumerate(k_values):
                # writer.writerow([metric_name, k, precision_values_by_metric[metric_name][i], map_values_by_metric[metric_name][i]])
                writer.writerow([metric_name, k, precision_values_by_metric[metric_name][i], map_values_by_metric[metric_name][i], f1_values_by_metric[metric_name][i]])

    print(f"Query results saved to {query_results_csv_path}")

    plot_precision_at_k(k_values, precision_values_by_metric, results_dir)
    plot_map_at_k(k_values, map_values_by_metric, results_dir)
    plot_f1_score_at_k(k_values, f1_values_by_metric, results_dir)
    
    print("All evaluations and plots completed.")

if __name__ == "__main__":
    main()
    