import json

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            for i, line in enumerate(c['source']):
                if "features = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal', " in line and '"mths_since_last_delinq_never"' in line:
                    c['source'][i] = "features = ['loan_amnt', 'int_rate', 'total_annual_inc', 'total_dti', 'revol_util', 'revol_bal']\n"
                    print("Fixed features in PCA cell")
                    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    main()
