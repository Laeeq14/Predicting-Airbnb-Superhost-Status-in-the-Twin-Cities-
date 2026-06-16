import pandas as pd, re
from collections import Counter

# Load both datasets
df_june = pd.read_csv('listings_detailed_june.csv', usecols=['amenities', 'host_is_superhost'], low_memory=False)
df_sep  = pd.read_csv('listings_new.csv', usecols=['amenities', 'host_is_superhost'], low_memory=False)
df = pd.concat([df_june, df_sep], ignore_index=True)
df = df.dropna(subset=['host_is_superhost', 'amenities'])
df['is_sh'] = df['host_is_superhost'].astype(str).str.strip().map({'t': 1, 'f': 0, '1': 1, '0': 0})
df = df.dropna(subset=['is_sh'])
df['is_sh'] = df['is_sh'].astype(int)

print(f"Total listings: {len(df)}, Superhost: {df['is_sh'].sum()}, Non-SH: {(df['is_sh']==0).sum()}")

# Parse amenities
def parse_amenities(x):
    items = re.findall(r'[\"\'](.*?)[\"\']', str(x))
    return set(i.strip().lower() for i in items if i.strip() and i.strip() != ',')

df['amenity_set'] = df['amenities'].apply(parse_amenities)

# Find amenities that appear in >= 5% of listings (meaningful signal)
all_counts = Counter()
for s in df['amenity_set']:
    all_counts.update(s)
min_count = len(df) * 0.05
candidate_amenities = [a for a, c in all_counts.items() if c >= min_count]
print(f"\nAmenities appearing in >=5% of listings: {len(candidate_amenities)}")

# For each candidate, compute superhost rate difference
results = []
sh_df  = df[df['is_sh'] == 1]
nsh_df = df[df['is_sh'] == 0]
n_sh  = len(sh_df)
n_nsh = len(nsh_df)

for am in candidate_amenities:
    sh_has  = sh_df['amenity_set'].apply(lambda s: am in s).sum()
    nsh_has = nsh_df['amenity_set'].apply(lambda s: am in s).sum()
    sh_rate  = sh_has  / n_sh  if n_sh  > 0 else 0
    nsh_rate = nsh_has / n_nsh if n_nsh > 0 else 0
    total_pct = all_counts[am] / len(df)
    results.append({
        'amenity': am,
        'sh_pct':  round(sh_rate * 100, 1),
        'nsh_pct': round(nsh_rate * 100, 1),
        'diff':    round((sh_rate - nsh_rate) * 100, 1),
        'total_pct': round(total_pct * 100, 1),
        'count': all_counts[am],
    })

results.sort(key=lambda x: abs(x['diff']), reverse=True)

print("\n=== TOP 40 AMENITIES BY SUPERHOST RATE DIFFERENCE ===")
print(f"{'Amenity':<35} {'SH%':>6} {'NonSH%':>7} {'Diff':>6} {'Total%':>7}")
print("-" * 65)
for r in results[:40]:
    print(f"{r['amenity']:<35} {r['sh_pct']:>6} {r['nsh_pct']:>7} {r['diff']:>+6} {r['total_pct']:>7}")
