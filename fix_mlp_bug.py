import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            
            # The offending leftover line from previous regex replacement
            old_test_line = "X_test_nn = X_test.drop(columns=[c for c in cat_cols if c in X_test.columns]).fillna(X_train.drop(columns=[c for c in cat_cols if c in X_train.columns]).median())"
            
            if old_test_line in source:
                # Remove it
                source = source.replace(old_test_line, "")
                
            c.source = source
            
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
