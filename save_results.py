import os
import csv
import pandas as pd

def save_in_csv(accuracy, nll, ece, model_name):
    
    # Save results to CSV
    csv_path = "./save_folder/save_results.csv"
    file_exists = os.path.isfile(csv_path)
    
    if file_exists:
        # Check if file is empty or has content
        is_empty = os.stat(csv_path).st_size == 0
        if is_empty:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Model', 'Accuracy', 'NLL', 'ECE'])
                writer.writerow([model_name, f"{accuracy:.2f}", f"{nll:.4f}", f"{ece:.4f}"])
        else:
            # File exists with content, append the new row
            df = pd.read_csv(csv_path)
            new_row = pd.DataFrame({'Model': [model_name], 
                                'Accuracy': [f"{accuracy:.2f}"], 
                                'NLL': [f"{nll:.4f}"],
                                'ECE': [f"{ece:.4f}"]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_path, index=False)
    else:
        # File doesn't exist, create it
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Accuracy', 'NLL', 'ECE'])
            writer.writerow([model_name, f"{accuracy:.2f}", f"{nll:.4f}", f"{ece:.4f}"])
    
    print(f"Results saved to {csv_path}")

    return