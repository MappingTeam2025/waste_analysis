import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('LL_final_transects.xlsx')
waste_columns = ['Organic ', 'Plastic', 'Paper', 'Hazardous', 'Glass/Metal', 'Burning Evidence']

def jaccard_similarity(data, var1, var2):
    """Calculate Jaccard similarity for binary variables"""
    both_present = ((data[var1] == 1) & (data[var2] == 1)).sum()
    either_present = ((data[var1] == 1) | (data[var2] == 1)).sum()
    if either_present == 0:
        return 0
    return both_present / either_present

def phi_coefficient(data, var1, var2):
    """Calculate Phi coefficient for binary variables"""
    contingency = pd.crosstab(data[var1], data[var2])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    phi = np.sqrt(chi2 / n)
    
    # Add sign based on positive/negative association
    if contingency.shape == (2, 2):
        a = contingency.iloc[0, 0]
        b = contingency.iloc[0, 1]
        c = contingency.iloc[1, 0]
        d = contingency.iloc[1, 1]
        phi_signed = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        return phi_signed, p_value
    return phi, p_value

def create_similarity_matrices(data, waste_cols):
    """Create Jaccard and Phi matrices"""
    n_vars = len(waste_cols)
    jaccard_matrix = np.zeros((n_vars, n_vars))
    phi_matrix = np.zeros((n_vars, n_vars))
    p_value_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(waste_cols):
        for j, var2 in enumerate(waste_cols):
            if i == j:
                jaccard_matrix[i, j] = 1.0
                phi_matrix[i, j] = 1.0
                p_value_matrix[i, j] = 0.0
            else:
                jaccard_matrix[i, j] = jaccard_similarity(data, var1, var2)
                phi, p_val = phi_coefficient(data, var1, var2)
                phi_matrix[i, j] = phi
                p_value_matrix[i, j] = p_val
    
    return jaccard_matrix, phi_matrix, p_value_matrix

communes = ['Hang Kia', 'Mai Hich']

for commune in communes:
    print(f"\n{'='*60}")
    print(f"Analysis for {commune}")
    print(f"{'='*60}\n")
    
    cd = df[df['Commune'] == commune].copy()
    jaccard, phi, p_values = create_similarity_matrices(cd, waste_columns)
    
    jaccard_df = pd.DataFrame(jaccard, index=waste_columns, columns=waste_columns)
    phi_df = pd.DataFrame(phi, index=waste_columns, columns=waste_columns)
    p_values_df = pd.DataFrame(p_values, index=waste_columns, columns=waste_columns)
    
    print(f"{commune} - Jaccard Similarity Matrix:")
    print(jaccard_df.round(3))
    print(f"\n{commune} - Phi Coefficient Matrix:")
    print(phi_df.round(3))
    print(f"\n{commune} - P-values:")
    print(p_values_df.round(4))
    
    # Create SEPARATE visualizations for Jaccard
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(jaccard_df, annot=True, fmt='.2f', cmap='Reds', 
                vmin=0, vmax=1, square=True, 
                cbar_kws={'label': "Jaccard Similarity"}, ax=ax1,
                linewidths=0.5, linecolor='white')
    ax1.set_title(f'{commune}: Jaccard Similarity Matrix', 
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{commune}_Jaccard_Similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create SEPARATE visualization for Phi Coefficient
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(phi_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                vmin=-1, vmax=1, center=0, square=True, 
                cbar_kws={'label': "Phi Coefficient"}, ax=ax2,
                linewidths=0.5, linecolor='white')
    ax2.set_title(f'{commune}: Phi Coefficient Matrix', 
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{commune}_Phi_Coefficient.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Excel files (with p-values included there)
    with pd.ExcelWriter(f'{commune}_jaccard_phi_results.xlsx') as writer:
        jaccard_df.to_excel(writer, sheet_name='Jaccard_Similarity')
        phi_df.to_excel(writer, sheet_name='Phi_Coefficient')
        p_values_df.to_excel(writer, sheet_name='P_values')
    
    print(f"\n✓ Results saved to {commune}_jaccard_phi_results.xlsx")
    print(f"✓ Jaccard visual saved to {commune}_Jaccard_Similarity.png")
    print(f"✓ Phi visual saved to {commune}_Phi_Coefficient.png\n")

print("="*60)
print("Analysis complete!")
print("="*60)