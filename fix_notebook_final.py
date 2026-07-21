import json

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for c in nb['cells']:
        if c['cell_type'] == 'markdown':
            for i, line in enumerate(c['source']):
                if "superiority (0.745 AUC) over the Logistic Regression baseline (0.712 AUC)" in line:
                    c['source'][i] = line.replace(
                        "LightGBM's superiority (0.745 AUC) over the Logistic Regression baseline (0.712 AUC)",
                        "LightGBM's strong predictive capability"
                    )
                    
                if "a Net Expected Profit of **$125,000,000** while maintaining an 82.5% approval rate." in line:
                    c['source'][i] = line.replace(
                        "a Net Expected Profit of **$125,000,000** while maintaining an 82.5% approval rate.",
                        "a maximized Net Expected Profit while maintaining a healthy approval rate."
                    )
                    
                if "underperforms the LightGBM model alone (0.7450)." in line:
                    c['source'][i] = line.replace(
                        "underperforms the LightGBM model alone (0.7450).",
                        "underperforms the LightGBM model alone (0.7272)."
                    )
                    
                if "**LightGBM + Optuna** | **0.7450** |" in line:
                    c['source'][i] = line.replace(
                        "**LightGBM + Optuna** | **0.7450** |",
                        "**LightGBM + Optuna** | **0.7272** |"
                    )
                    
                if "down to the optimal 0.18, we correctly" in line:
                    c['source'][i] = line.replace(
                        "down to the optimal 0.18, we correctly",
                        "down to the optimal point, we correctly"
                    )

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook markdown text updated to remove unverifiable claims.")

if __name__ == "__main__":
    main()
