import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            
            # Fix features list: replace annual_inc with total_annual_inc, dti with total_dti
            # Add missingness flags
            if "features =" in source and "loan_amnt" in source:
                old_feat1 = '"annual_inc"'
                if old_feat1 in source:
                    source = source.replace(old_feat1, '"total_annual_inc"')
                old_feat1_sq = "'annual_inc'"
                if old_feat1_sq in source:
                    source = source.replace(old_feat1_sq, "'total_annual_inc'")
                    
                old_feat2 = '"dti"'
                if old_feat2 in source:
                    source = source.replace(old_feat2, '"total_dti"')
                old_feat2_sq = "'dti'"
                if old_feat2_sq in source:
                    source = source.replace(old_feat2_sq, "'total_dti'")
                    
                # Add missingness flags to the features list if not already there
                if "mths_since_last_delinq_never" not in source and "features =" in source:
                    # We will append them right before the closing bracket of features = [...]
                    # This is tricky with regex, let's just do a string replace if possible
                    # We know the block ends with ']'.
                    # Let's find the assignment of features = [...]
                    import re
                    match = re.search(r'features\s*=\s*\[(.*?)\]', source, re.DOTALL)
                    if match:
                        inner = match.group(1)
                        if "mths_since_last_delinq_never" not in inner:
                            new_inner = inner + ', "mths_since_last_delinq_never", "mths_since_last_record_never", "mths_since_last_major_derog_never", "mths_since_recent_bc_dlq_never", "mths_since_recent_revol_delinq_never", "mths_since_recent_inq_never"'
                            source = source.replace(inner, new_inner)
                            
            c.source = source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
