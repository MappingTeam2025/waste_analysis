import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel('LL_final_transects.xlsx')

# Waste type columns (binary presence/absence)
waste_columns = ['Organic ', 'Plastic', 'Paper', 'Hazardous', 'Glass/Metal', 'Burning Evidence']

communes = ['Hang Kia', 'Mai Hich']

print("\n" + "="*80)
print("MANN-WHITNEY U TESTS: WASTE TYPES BY OPEN DUMPING PRESENCE")
print("="*80)

for commune in communes:
    print(f"\n{'='*80}")
    print(f"{commune.upper()} - OPEN DUMPING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Filter data for commune
    commune_data = df[df['Commune'] == commune].copy()
    
    # Check if 'Open Dumping' column exists
    if 'Open Dumping' not in commune_data.columns:
        print(f"Warning: 'Open Dumping' column not found for {commune}")
        continue
    
    # Split data by open dumping presence
    has_dumping = commune_data[commune_data['Open Dumping'] == 1]
    no_dumping = commune_data[commune_data['Open Dumping'] == 0]
    
    print(f"Transects WITH open dumping: {len(has_dumping)}")
    print(f"Transects WITHOUT open dumping: {len(no_dumping)}\n")
    
    # Results storage
    results = []
    
    for waste_type in waste_columns:
        # Get waste presence for each group
        with_dump = has_dumping[waste_type].dropna()
        without_dump = no_dumping[waste_type].dropna()
        
        # Skip if insufficient data
        if len(with_dump) < 2 or len(without_dump) < 2:
            print(f"{waste_type}: Insufficient data")
            continue
        
        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(with_dump, without_dump, alternative='two-sided')
        
        # Calculate medians
        median_with = with_dump.median()
        median_without = without_dump.median()
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Extremely significant"
        elif p_value < 0.01:
            interpretation = "Highly significant"
        elif p_value < 0.05:
            interpretation = "Significant"
        elif p_value < 0.10:
            interpretation = "Marginally significant"
        else:
            interpretation = "Not significant"
        
        results.append({
            'Waste Type': waste_type,
            'U Statistic': u_stat,
            'p-value': p_value,
            'Median (With Dumping)': median_with,
            'Median (Without Dumping)': median_without,
            'Interpretation': interpretation
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save to Excel
    results_df.to_excel(f'{commune}_OpenDumping_MannWhitney.xlsx', index=False)
    print(f"\n✓ Results saved to {commune}_OpenDumping_MannWhitney.xlsx")

print("\n" + "="*80)
print("MANN-WHITNEY U TESTS: WASTE TYPES BY BURNING EVIDENCE PRESENCE")
print("="*80)

for commune in communes:
    print(f"\n{'='*80}")
    print(f"{commune.upper()} - BURNING EVIDENCE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Filter data for commune
    commune_data = df[df['Commune'] == commune].copy()
    
    # Split data by burning evidence presence
    has_burning = commune_data[commune_data['Burning Evidence'] == 1]
    no_burning = commune_data[commune_data['Burning Evidence'] == 0]
    
    print(f"Transects WITH burning evidence: {len(has_burning)}")
    print(f"Transects WITHOUT burning evidence: {len(no_burning)}\n")
    
    # Results storage
    results = []
    
    # Test all waste types EXCEPT Burning Evidence (can't compare it to itself)
    test_columns = [col for col in waste_columns if col != 'Burning Evidence']
    
    for waste_type in test_columns:
        # Get waste presence for each group
        with_burn = has_burning[waste_type].dropna()
        without_burn = no_burning[waste_type].dropna()
        
        # Skip if insufficient data
        if len(with_burn) < 2 or len(without_burn) < 2:
            print(f"{waste_type}: Insufficient data")
            continue
        
        # Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(with_burn, without_burn, alternative='two-sided')
        
        # Calculate medians
        median_with = with_burn.median()
        median_without = without_burn.median()
        
        # Interpretation
        if p_value < 0.001:
            interpretation = "Extremely significant"
        elif p_value < 0.01:
            interpretation = "Highly significant"
        elif p_value < 0.05:
            interpretation = "Significant"
        elif p_value < 0.10:
            interpretation = "Marginally significant"
        else:
            interpretation = "Not significant"
        
        results.append({
            'Waste Type': waste_type,
            'U Statistic': u_stat,
            'p-value': p_value,
            'Median (With Burning)': median_with,
            'Median (Without Burning)': median_without,
            'Interpretation': interpretation
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save to Excel
    results_df.to_excel(f'{commune}_BurningEvidence_MannWhitney.xlsx', index=False)
    print(f"\n✓ Results saved to {commune}_BurningEvidence_MannWhitney.xlsx")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)