import json

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            for i, line in enumerate(c['source']):
                if 'corr_features = [' in line and 'mths_since_last_delinq_never' in line:
                    c['source'][i] = "corr_features = ['loan_amnt', 'int_rate', 'installment', 'total_annual_inc', 'total_dti', 'fico_range_low', 'revol_util']\n"
                    print("Fixed corr_features")
                elif 'mths_since_last_delinq_never' in line and ', "mths_since_last_delinq_never", "mths_since_last_record_never", "mths_since_last_major_derog_never", "mths_since_recent_bc_dlq_never", "mths_since_recent_revol_delinq_never", "mths_since_recent_inq_never"' in line:
                     c['source'][i] = line.replace(', "mths_since_last_delinq_never", "mths_since_last_record_never", "mths_since_last_major_derog_never", "mths_since_recent_bc_dlq_never", "mths_since_recent_revol_delinq_never", "mths_since_recent_inq_never"', '')
                     print("Fixed generic _never inclusion")
                     
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    main()
