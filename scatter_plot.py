import fire
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    """Parse le fichier texte et extrait Validation Accuracy et ECE par Epoch."""
    print(f"Parsing du fichier : {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    val_accuracies = []
    eces = []

    for line in lines:
        parts = line.strip().split(" | ")  # Séparer chaque colonne
        if len(parts) < 7:
            continue  # Ignorer les lignes mal formatées
        
        try:
            val_acc = float(parts[3].split(": ")[1])  # Validation Accuracy en 4e colonne
            ece = float(parts[4].split(": ")[1])  # ECE en 5e colonne
            
            val_accuracies.append(val_acc)
            eces.append(ece)
        except (ValueError, IndexError):
            continue  # Ignorer les lignes mal formatées
    
    if not val_accuracies or not eces:
        print("Aucune donnée trouvée dans le fichier.")
    return val_accuracies, eces

def plot_val_acc_vs_ece(file_path="mixup_model/resnet34/resnet34_4757.out", save_folder="mixup_model/plot_mixup"):
    """Trace un scatter plot Val Acc vs. ECE et enregistre en PDF."""
    print("Début de l'exécution de la fonction plot_val_acc_vs_ece")
    
    # Vérifier si le fichier existe
    try:
        val_accuracies, eces = parse_log_file(file_path)
    except Exception as e:
        print(f"Erreur lors du parsing du fichier : {e}")
        return
    
    if not val_accuracies or not eces:
        print("Les données sont vides, aucun graphique à générer.")
        return
    
    # Générer le nom du fichier de sortie
    file_name = file_path.split("/")[-1].split(".")[0]  # Récupérer le nom sans extension
    output_filename = f"scatter_{file_name}.pdf"
    output_pdf = f"{save_folder}/{output_filename}"

    # Création du scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(val_accuracies, eces, c="blue", marker="o", alpha=0.7)
    plt.xlabel("Validation Accuracy")
    plt.ylabel("ECE (Expected Calibration Error)")
    plt.title("Validation Accuracy vs. ECE")
    plt.grid(True)

    # Sauvegarde en PDF
    try:
        plt.savefig(output_pdf, format="pdf")
        print(f"✅ Scatter plot enregistré dans {output_pdf}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du graphique : {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    fire.Fire(plot_val_acc_vs_ece)
