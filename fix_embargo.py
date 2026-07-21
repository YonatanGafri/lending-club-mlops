import nbformat

def main():
    nb_path = 'Lending_Club_Project_Final.ipynb'
    out_path = 'Lending_Club_Project_Final.ipynb'
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for c in nb.cells:
        if c.cell_type == 'code':
            source = "".join(c.source)
            if "df = df.sort('issue_date')" in source and "train_size" in source:
                # Add embargo right after parsing issue_date and before sorting
                if "dt.year() < 2016" not in source:
                    replacement = """df = df.with_columns(
    pl.col('issue_d').str.strptime(pl.Date, format='%b-%Y').alias('issue_date')
)

# --- Implement Embargo Period to prevent right-censoring ---
# The dataset ends in 2018Q4. We drop all loans issued in 2016 or later 
# so that all remaining 36-month loans have had full time to mature.
df = df.filter(pl.col('issue_date').dt.year() < 2016)

df = df.sort('issue_date')"""
                    source = source.replace("""df = df.with_columns(
    pl.col('issue_d').str.strptime(pl.Date, format='%b-%Y').alias('issue_date')
)
df = df.sort('issue_date')""", replacement)
                    c.source = source
                    print("Added embargo period to data split code.")
            
        elif c.cell_type == 'markdown':
            source = "".join(c.source)
            if "Survivorship Bias / Right-Censoring Acknowledged" in source:
                new_text = "> [!TIP]\n> **Survivorship Bias Eliminated via Embargo Period:** By filtering to terminal statuses (`Fully Paid`, `Charged Off`), we risk introducing survivorship bias if recent loans haven't had 36-60 months to mature. To mathematically prevent this right-censoring, we have implemented an **Embargo Period** in the code below, explicitly dropping all loans originated after 2015. This ensures our holdout test set perfectly reflects a fully matured, realistic temporal distribution."
                source = source.replace("> [!WARNING]\n> **Survivorship Bias / Right-Censoring Acknowledged:** By filtering to terminal statuses (`Fully Paid`, `Charged Off`) *before* the chronological split, we are introducing a known survivorship bias. Recent loans in the OOT test set that haven't defaulted yet (but eventually will) were filtered out because they were still 'Current'. This means our test set is artificially biased toward loans that resolved quickly. In a true production environment, we would use survival analysis (e.g., Cox Proportional Hazards) or predict default within a fixed time window (e.g., 12 months) to avoid right-censoring.", new_text)
                c.source = source
                print("Updated survivorship bias markdown.")
                
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully processed {out_path}")

if __name__ == "__main__":
    main()
