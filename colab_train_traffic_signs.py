import json
import warnings
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")


DATA_ZIP_PATH = "/content/data_path.zip"
EXTRACT_PATH = "/content"
RAW_DATASET_PATH = "/content/traffic_Data/DATA"
MERGED_DATASET_PATH = "/content/traffic_Data_MERGED/DATA"
LABELS_PATH = "/content/labels.csv"
CLASS_ORDER_PATH = "/content/class_order.json"
MODEL_PATH = "/content/traffic_sign_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS_STAGE_1 = 12
EPOCHS_STAGE_2 = 10
INITIAL_LR = 3e-4
FINE_TUNE_LR = 5e-5


def extract_dataset():
    with ZipFile(DATA_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)


def load_datasets(dataset_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def save_class_order(class_names):
    with open(CLASS_ORDER_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file)


def compute_class_weights(class_names, dataset_path):
    dataset_root = Path(dataset_path)
    counts = {}

    for index, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        image_count = sum(
            1
            for path in class_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        counts[index] = max(image_count, 1)

    max_count = max(counts.values())
    class_weights = {
        index: min(max_count / count, 4.0)
        for index, count in counts.items()
    }

    return class_weights, counts


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def compile_model(model, learning_rate):
    try:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05)
    except TypeError:
        # Older TensorFlow versions don't support label_smoothing here.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )


def plot_history(histories):
    history = {}
    for item in histories:
        for key, values in item.history.items():
            history.setdefault(key, []).extend(values)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_speed_limit_counts(labels_df, class_names, class_counts):
    names_by_class = {
        str(class_id): name
        for class_id, name in zip(labels_df["ClassId"], labels_df["Name"])
    }

    print("\nSpeed-limit class counts:")
    for class_name in class_names:
        label = names_by_class.get(class_name, class_name)
        if "speed limit" in label.lower():
            print(
                f"  {class_name}: {label} -> {class_counts[class_names.index(class_name)]} images"
            )


def merge_duplicate_classes(labels_df, source_root, target_root):
    source_root = Path(source_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    name_to_target = {}
    remap = {}

    for class_id, name in zip(labels_df["ClassId"], labels_df["Name"]):
        class_id = str(class_id)
        if name not in name_to_target:
            name_to_target[name] = class_id
        remap[class_id] = name_to_target[name]

    # Copy files into merged folders
    for class_dir in source_root.iterdir():
        if not class_dir.is_dir():
            continue
        src_class = class_dir.name
        if src_class not in remap:
            continue
        dst_class = remap[src_class]
        dst_dir = target_root / dst_class
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            dst_path = dst_dir / img_path.name
            if dst_path.exists():
                # Avoid overwriting if names collide
                dst_path = dst_dir / f"{img_path.stem}_{src_class}{img_path.suffix}"
            dst_path.write_bytes(img_path.read_bytes())

    # Build merged labels map for reference
    merged_labels = {}
    for class_id, name in zip(labels_df["ClassId"], labels_df["Name"]):
        merged_labels[remap[str(class_id)]] = name

    return remap, merged_labels


def main():
    labels_df = pd.read_csv(LABELS_PATH)
    print(labels_df.head())

    extract_dataset()
    remap, merged_labels = merge_duplicate_classes(
        labels_df, RAW_DATASET_PATH, MERGED_DATASET_PATH
    )
    print("Merged class mapping (sample):", list(remap.items())[:10])

    train_ds, val_ds, class_names = load_datasets(MERGED_DATASET_PATH)
    save_class_order(class_names)

    class_weights, class_counts = compute_class_weights(class_names, MERGED_DATASET_PATH)
    print("Model class order:", class_names)
    print("Class weights:", class_weights)
    show_speed_limit_counts(labels_df, class_names, class_counts)

    model, backbone = build_model(num_classes=len(class_names))
    compile_model(model, INITIAL_LR)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
    )

    history_stage_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
    )

    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False

    compile_model(model, FINE_TUNE_LR)

    history_stage_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1 + EPOCHS_STAGE_2,
        initial_epoch=len(history_stage_1.history["loss"]),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
    )

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Class order saved to {CLASS_ORDER_PATH}")

    plot_history([history_stage_1, history_stage_2])


if __name__ == "__main__":
    main()
