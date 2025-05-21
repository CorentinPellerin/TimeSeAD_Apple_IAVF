from apple_dataset import AppleDataset

# Ta liste complète
forced = [67, 20, 27, 100, 86, 61, 76, 37, 53, 36, 60, 85, 45, 56, 35, 91,
           32, 118, 121, 123, 46, 119, 120, 90, 9, 122, 82]

# Instancie ton dataset en test uniquement
ds = AppleDataset(
    dataset_path="/home/pellerinc/TimeSeAD-KFI/data/AppleDataset",
    training=False,
    forced_test_ids=forced,
    preprocess=True  # si tu as déjà pré‐traité
)

# Affiche la liste exacte des fichiers chargés
print(ds.test_files_basenames)

