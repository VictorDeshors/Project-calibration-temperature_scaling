import fire
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    """Parse le fichier texte et extrait les données."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    data = {}
    
    for line in lines:
        parts = line.strip().split(" | ")  # Séparer chaque colonne
        if len(parts) < 7:
            continue  # Ignorer les lignes mal formatées
        
        try:
            epoch = int(parts[0].split(" ")[1].split("/")[0])  # Extraire l'epoch
            values = [float(part.split(": ")[1]) for part in parts[1:]]  # Extraire les valeurs numériques
            epochs.append(epoch)
            
            # Définir les noms des colonnes
            keys = ["Train Loss", "Train Acc", "Val Acc", "ECE", "Adaptive ECE", "OE"]
            for i, key in enumerate(keys):
                if key not in data:
                    data[key] = []
                data[key].append(values[i])
        except (ValueError, IndexError):
            continue  # Ignorer les lignes mal formatées
    
    return epochs, data

def plot_and_save(file_path, save_folder="mixup_model/plot_mixup"):
    """Trace les courbes sur 3 sous-graphes différents et enregistre en PDF."""
    # Générer le nom du fichier de sortie
    file_name = file_path.split("/")[-1].split(".")[0]  # Récupérer le nom sans extension
    output_filename = f"plot_{file_name}.pdf"
    output_pdf = save_folder + "/" + output_filename
    
    # Extraire les données du fichier
    epochs, data = parse_log_file(file_path)
    
    # Diviser les métriques en 2 groupes de 3
    first_group = ["Train Loss", "Train Acc", "Val Acc"]
    second_group = ["ECE", "Adaptive ECE", "OE"]
    
    # Création de la figure avec 3 sous-graphes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    
    # Tracer le premier groupe (3 premières métriques)
    ax1 = axes[0]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metrics")
    
    for key in first_group:
        if key in data:
            ax1.plot(epochs, data[key], label=key, linestyle="-", marker="o")
    
    ax1.set_title("First metrics: Train Loss, Train Acc, Val Acc")
    ax1.grid(True)
    ax1.legend()
    
    # Tracer le deuxième groupe (3 dernières métriques)
    ax2 = axes[1]
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metrics")
    
    for key in second_group:
        if key in data:
            ax2.plot(epochs, data[key], label=key, linestyle="-", marker="o")
    
    ax2.set_title("Last metrics: ECE, Adaptive ECE, OE")
    ax2.grid(True)
    ax2.legend()
    
    # Ajout d'un troisième graphique pour ECE vs Val Acc
    ax3 = axes[2]
    ax3.set_xlabel("ECE")
    ax3.set_ylabel("Validation Accuracy (Val Acc)")
    
    if "ECE" in data and "Val Acc" in data:
        # Créer un gradient de couleurs pour montrer la progression des epochs
        colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Ajouter des points pour chaque epoch avec gradient de couleur (sans annotations)
        scatter = ax3.scatter(data["ECE"], data["Val Acc"], c=epochs, cmap='viridis', s=60)
        
        # Ajouter une légende colorée pour les epochs
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Epoch')
        
        # Tracer une ligne pour montrer l'évolution
        ax3.plot(data["ECE"], data["Val Acc"], color='gray', alpha=0.5, linestyle='-')
        
        # Ajouter une bissectrice
        # Déterminer les limites min/max pour la bissectrice
        all_values = data["ECE"] + data["Val Acc"]
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Tracer la bissectrice y=x
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label="Bissectrice (y=x)", alpha=0.7)
        ax3.legend()
    
    ax3.set_title("Val Acc vs ECE (/epoch)")
    ax3.grid(True)
    
    # Sauvegarde en PDF
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    print(f"✅ Courbes enregistrées dans {output_pdf}")
    
    # Créer également une version PNG pour faciliter la visualisation
    output_png = save_folder + "/" + f"plot_{file_name}.png"
    plt.savefig(output_png, format="png", dpi=300)
    print(f"✅ Version PNG également enregistrée dans {output_png}")
    
    plt.close()

if __name__ == "__main__":
    fire.Fire(plot_and_save)