import json
import os

# Chemin du fichier de configuration original
original_config_path = 'mlpmixer.json'

# Liste des tailles de patch à tester
patch_sizes = [4, 8, 16]

# Crée les fichiers de configuration pour chaque taille de patch
for patch_size in patch_sizes:
    # Charge le fichier de configuration original
    with open(original_config_path, 'r') as f:
        config = json.load(f)
    
    # Modifie la valeur de patch_size
    config['patch_size'] = patch_size
    
    # Chemin du nouveau fichier de configuration
    new_config_path = f'mlpmixer_patch_{patch_size}.json'
    
    # Sauvegarde le nouveau fichier de configuration
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Fichier de configuration créé : {new_config_path}")