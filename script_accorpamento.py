import os
import numpy as np
import shutil
from tqdm import tqdm

def reorganize_cityscapes_dataset(input_root_dir, output_root_dir):
    """
    Riorganizza un dataset Cityscapes accorpando alcune classi nelle maschere di segmentazione.

    Args:
        input_root_dir (str): Percorso della directory radice del dataset Cityscapes originale.
        output_root_dir (str): Percorso della directory radice dove verrà salvato il nuovo dataset.
    """

    # Mappatura delle classi originali alle nuove classi accorpate
    # Le classi non specificate qui verranno copiate con il loro ID originale.
    class_mapping = {
        # Vehicle: 100 (un nuovo ID arbitrario, assicurarsi che non si sovrapponga a classi esistenti se non ri-mappate)
        13: 3,  # car
        14: 3,  # truck
        15: 3,  # bus
        16: 3,  # train
        17: 3,  # motorcycle
        18: 3,  # bicycle

        # Human: 101
        11: 4,  # person
        12: 4,  # rider

        # Street Furniture: 102
        5: 5,   # pole
        6: 5,   # traffic light
        7: 5,   # traffic sign

        # Structure: 103
        2: 6,   # building
        3: 6,   # wall
        4: 6,   # fence

        # Nature: 104
        8: 7,   # vegetation
        9: 7,   # terrain
        #10: 104,  # sky

        #reorder
        10:2 #sky,

    }

    # Definisci le nuove etichette per chiarezza (per informazione)
    new_class_labels = {
        0: 'road',
        1: 'sidewalk',
        2:'sky',
        3: 'vehicle',
        4: 'human',
        5: 'street_furniture',
        6: 'structure',
        7: 'nature',
        # Potrei aggiungere qui altre classi che mantengono il loro ID originale per un elenco completo
    }

    subdirs = ['train', 'val', 'test']
    data_types = ['image', 'depth', 'label'] # Ordine per iterazione

    for subdir in subdirs:
        print(f"Elaborazione della sottocartella: {subdir}")
        input_subdir_path = os.path.join(input_root_dir, subdir)
        output_subdir_path = os.path.join(output_root_dir, subdir)

        # Crea le directory di output se non esistono
        for data_type in data_types:
            os.makedirs(os.path.join(output_subdir_path, data_type), exist_ok=True)

        # Elabora le immagini
        print(f"Copia immagini per {subdir}...")
        input_images_path = os.path.join(input_subdir_path, 'image')
        output_images_path = os.path.join(output_subdir_path, 'image')
        for img_file in tqdm(os.listdir(input_images_path)):
            shutil.copy(os.path.join(input_images_path, img_file), output_images_path)

        # Elabora le mappe di profondità
        print(f"Copia mappe di profondità per {subdir}...")
        input_depth_path = os.path.join(input_subdir_path, 'depth')
        output_depth_path = os.path.join(output_subdir_path, 'depth')
        for depth_file in tqdm(os.listdir(input_depth_path)):
            shutil.copy(os.path.join(input_depth_path, depth_file), output_depth_path)

        # Elabora le maschere di segmentazione
        print(f"Elaborazione maschere di segmentazione per {subdir}...")
        input_labels_path = os.path.join(input_subdir_path, 'label')
        output_labels_path = os.path.join(output_subdir_path, 'label')
        for label_file in tqdm(os.listdir(input_labels_path)):
            if label_file.endswith('.npy'):
                input_label_path = os.path.join(input_labels_path, label_file)
                output_label_path = os.path.join(output_labels_path, label_file)

                mask = np.load(input_label_path)
                processed_mask = np.copy(mask) # Lavora su una copia per non modificare l'originale

                # Applica la mappatura delle classi
                for original_id, new_id in class_mapping.items():
                    processed_mask[mask == original_id] = new_id

                np.save(output_label_path, processed_mask)

    print("\nRiorganizzazione del dataset completata!")
    print(f"Il nuovo dataset è stato salvato in: {output_root_dir}")

if __name__ == "__main__":
    # Imposta i percorsi del tuo dataset
    input_dataset_path = 'C:\\Users\\perri\\Downloads\\cityscapes'  # Sostituire con il percorso del dataset originale
    output_dataset_path = 'C:\\Users\\perri\\Downloads\\new_cityscapes_dataset' # Sostituire con il percorso dove vuoi salvare il nuovo dataset

    reorganize_cityscapes_dataset(input_dataset_path, output_dataset_path)