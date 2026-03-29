"""
Generate publication-quality validation figures comparing real HMDA vs synthetic data results.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load results
with open('/sessions/elegant-busy-planck/mnt/causal -fairness/output/results.json') as f:
    synthetic = json.load(f)
with open('/sessions/elegant-busy-planck/mnt/causal -fairness/output/real_hmda_results.json') as f:
    real = json.load(f)

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'synthetic': '#2196F3',
    'real': '#FF5722',
    'ide': '#E53935',
    'iie': '#1E88E5',
    'white': '#4CAF50',
    'black': '#FF9800',
}

outdir = '/sessions/elegant-busy-planck/mnt/causal -fairness/figures'

# ============================================================
# FIGURE 5: Real vs Synthetic Comparison Dashboard
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
fig.suptitle('Empirical Validation: Real HMDA (NY 2022) vs Synthetic Data',
             fontsize=16, fontweight='bold', y=0.98)

# Panel A: Denial Rates by Race
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(2)
width = 0.35
bars1 = ax1.bar(x - width/2,
                [synthetic['white_denial_rate']*100, synthetic['black_denial_rate']*100],
                width, label='Synthetic', color=COLORS['synthetic'], alpha=0.8, edgecolor='white')
bars2 = ax1.bar(x + width/2,
                [real['white_denial_rate']*100, real['black_denial_rate']*100],
                width, label='Real HMDA', color=COLORS['real'], alpha=0.8, edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(['White', 'Black'])
ax1.set_ylabel('Denial Rate (%)')
ax1.set_title('(A) Denial Rates by Race')
ax1.legend()
ax1.set_ylim(0, 22)
# Add value labels
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

# Panel B: Decomposition Comparison
ax2 = fig.add_subplot(gs[0, 1])
categories = ['Total\nEffect', 'IDE\n(Direct)', 'IIE\n(Indirect)']
synth_vals = [synthetic['total_effect_pp'], synthetic['ide_pp'], synthetic['iie_pp']]
real_vals = [real['total_effect_pp'], real['ide_pp'], real['iie_pp']]
synth_ci_lo = [0, synthetic['ide_ci_lower'], synthetic['iie_ci_lower']]
synth_ci_hi = [0, synthetic['ide_ci_upper'], synthetic['iie_ci_upper']]
real_ci_lo = [0, real['ide_ci_lower'], real['iie_ci_lower']]
real_ci_hi = [0, real['ide_ci_upper'], real['iie_ci_upper']]

x = np.arange(3)
synth_err = [[v - lo for v, lo in zip(synth_vals, synth_ci_lo)],
             [hi - v for v, hi in zip(synth_vals, synth_ci_hi)]]
real_err = [[v - lo for v, lo in zip(real_vals, real_ci_lo)],
            [hi - v for v, hi in zip(real_vals, real_ci_hi)]]
# No CI for TE
synth_err[0][0] = synth_err[1][0] = 0
real_err[0][0] = real_err[1][0] = 0

ax2.bar(x - width/2, synth_vals, width, yerr=synth_err, capsize=4,
        label='Synthetic', color=COLORS['synthetic'], alpha=0.8, edgecolor='white')
ax2.bar(x + width/2, real_vals, width, yerr=real_err, capsize=4,
        label='Real HMDA', color=COLORS['real'], alpha=0.8, edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Effect (percentage points)')
ax2.set_title('(B) Causal Decomposition')
ax2.legend()
ax2.set_ylim(0, 12)
for i, (sv, rv) in enumerate(zip(synth_vals, real_vals)):
    ax2.text(i - width/2, sv + synth_err[1][i] + 0.2, f'{sv:.1f}', ha='center', fontsize=8)
    ax2.text(i + width/2, rv + real_err[1][i] + 0.2, f'{rv:.1f}', ha='center', fontsize=8)

# Panel C: IDE/IIE Proportions (Stacked)
ax3 = fig.add_subplot(gs[0, 2])
categories_prop = ['Synthetic\nData', 'Real HMDA\n(NY 2022)']
ide_pcts = [synthetic['ide_pct_of_te'], real['ide_pct_of_te']]
iie_pcts = [synthetic['iie_pct_of_te'], real['iie_pct_of_te']]
x = np.arange(2)
width_prop = 0.5
ax3.bar(x, ide_pcts, width_prop, label='IDE (Direct Discrimination)',
        color=COLORS['ide'], alpha=0.85)
ax3.bar(x, iie_pcts, width_prop, bottom=ide_pcts,
        label='IIE (Structural Inequality)', color=COLORS['iie'], alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(categories_prop)
ax3.set_ylabel('% of Total Effect')
ax3.set_title('(C) Decomposition Proportions')
ax3.legend(loc='upper right', fontsize=8)
ax3.set_ylim(0, 110)
for i in range(2):
    ax3.text(i, ide_pcts[i]/2, f'{ide_pcts[i]:.1f}%', ha='center', va='center',
             fontweight='bold', color='white', fontsize=10)
    ax3.text(i, ide_pcts[i] + iie_pcts[i]/2, f'{iie_pcts[i]:.1f}%', ha='center', va='center',
             fontweight='bold', color='white', fontsize=10)

# Panel D: Path-Specific Effects
ax4 = fig.add_subplot(gs[1, 0])
path_names = ['DTI', 'Credit Score', 'Income', 'LTV']
synth_paths = [synthetic['path_dti_numeric_pp'], synthetic['path_credit_score_quintile_pp'],
               synthetic['path_income_quintile_pp'], synthetic['path_ltv_pp']]
real_paths = [real['path_effects']['dti_numeric']['effect_pp'],
              real['path_effects']['credit_score_quintile']['effect_pp'],
              real['path_effects']['income_quintile']['effect_pp'],
              real['path_effects']['ltv']['effect_pp']]

x = np.arange(4)
ax4.barh(x + width/2, synth_paths, width, label='Synthetic',
         color=COLORS['synthetic'], alpha=0.8)
ax4.barh(x - width/2, real_paths, width, label='Real HMDA',
         color=COLORS['real'], alpha=0.8)
ax4.set_yticks(x)
ax4.set_yticklabels(path_names)
ax4.set_xlabel('Indirect Effect (pp)')
ax4.set_title('(D) Path-Specific Indirect Effects')
ax4.legend(loc='lower right')
ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Panel E: E-value Sensitivity Comparison
ax5 = fig.add_subplot(gs[1, 1])
gamma_range = np.linspace(1, 4, 200)
# Bias factor for E-value: at gamma, the observed RR could be reduced by gamma + sqrt(gamma*(gamma-1))
# IDE is nullified when confounding bias >= observed IDE RR
synth_rr = (synthetic['white_denial_rate'] + synthetic['ide_pp']/100) / synthetic['white_denial_rate']
real_rr = (real['white_denial_rate'] + real['ide_pp']/100) / real['white_denial_rate']

# Sensitivity curves
synth_bias = gamma_range + np.sqrt(gamma_range * (gamma_range - 1))
real_bias = gamma_range + np.sqrt(gamma_range * (gamma_range - 1))

ax5.axhline(y=synth_rr, color=COLORS['synthetic'], linestyle='--', alpha=0.7, label=f'Synthetic IDE RR={synth_rr:.2f}')
ax5.axhline(y=real_rr, color=COLORS['real'], linestyle='--', alpha=0.7, label=f'Real IDE RR={real_rr:.2f}')
ax5.plot(gamma_range, synth_bias, color='gray', alpha=0.5, label='Confounding bias')
ax5.fill_between(gamma_range, 0, synth_bias, alpha=0.05, color='gray')

# Mark E-values
ax5.plot(synthetic['e_value_point'], synth_rr, 'o', color=COLORS['synthetic'], markersize=10, zorder=5)
ax5.plot(real['e_value_point'], real_rr, 'o', color=COLORS['real'], markersize=10, zorder=5)
ax5.annotate(f"E={synthetic['e_value_point']:.2f}",
             (synthetic['e_value_point'], synth_rr),
             textcoords="offset points", xytext=(10, 10), fontsize=9)
ax5.annotate(f"E={real['e_value_point']:.2f}",
             (real['e_value_point'], real_rr),
             textcoords="offset points", xytext=(10, -15), fontsize=9)

ax5.set_xlabel('Confounding strength (Γ)')
ax5.set_ylabel('Risk Ratio')
ax5.set_title('(E) E-value Sensitivity Analysis')
ax5.legend(fontsize=8)
ax5.set_xlim(1, 4)
ax5.set_ylim(0.8, 5)

# Panel F: Summary Statistics Table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [
    ['Metric', 'Synthetic', 'Real HMDA'],
    ['N (total)', f"{synthetic['n_total']:,}", f"{real['n_total']:,}"],
    ['White denial', f"{synthetic['white_denial_rate']*100:.1f}%", f"{real['white_denial_rate']*100:.1f}%"],
    ['Black denial', f"{synthetic['black_denial_rate']*100:.1f}%", f"{real['black_denial_rate']*100:.1f}%"],
    ['TE (pp)', f"{synthetic['total_effect_pp']:.2f}", f"{real['total_effect_pp']:.2f}"],
    ['IDE (pp)', f"{synthetic['ide_pp']:.2f}", f"{real['ide_pp']:.2f}"],
    ['IIE (pp)', f"{synthetic['iie_pp']:.2f}", f"{real['iie_pp']:.2f}"],
    ['IDE %', f"{synthetic['ide_pct_of_te']:.1f}%", f"{real['ide_pct_of_te']:.1f}%"],
    ['IIE %', f"{synthetic['iie_pct_of_te']:.1f}%", f"{real['iie_pct_of_te']:.1f}%"],
    ['E-value', f"{synthetic['e_value_point']:.2f}", f"{real['e_value_point']:.2f}"],
]
table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)
# Style header
for j in range(3):
    table[0, j].set_facecolor('#37474F')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Alternate row colors
for i in range(1, len(table_data)):
    color = '#F5F5F5' if i % 2 == 0 else 'white'
    for j in range(3):
        table[i, j].set_facecolor(color)
ax6.set_title('(F) Summary Comparison', pad=20)

plt.savefig(f'{outdir}/figure5_real_vs_synthetic_validation.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/figure5_real_vs_synthetic_validation.pdf', bbox_inches='tight')
print("Saved figure5_real_vs_synthetic_validation.png/pdf")

# ============================================================
# FIGURE 6: Real HMDA Decomposition Bar Chart (standalone)
# ============================================================
fig2, ax = plt.subplots(figsize=(8, 6))
categories = ['Total Effect\n(TE)', 'Direct Effect\n(IDE)', 'Indirect Effect\n(IIE)']
values = [real['total_effect_pp'], real['ide_pp'], real['iie_pp']]
colors = ['#546E7A', COLORS['ide'], COLORS['iie']]
ci_lower = [0, real['ide_ci_lower'], real['iie_ci_lower']]
ci_upper = [0, real['ide_ci_upper'], real['iie_ci_upper']]
errors = [[v - lo for v, lo in zip(values, ci_lower)],
          [hi - v for v, hi in zip(values, ci_upper)]]
errors[0][0] = errors[1][0] = 0

bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='white',
              linewidth=1.5, yerr=errors, capsize=6, error_kw={'linewidth': 2})

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.2f} pp', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Effect Size (percentage points)', fontsize=13)
ax.set_title('Causal Decomposition of Racial Disparity\nReal HMDA Data — NY 2022, Conventional Home Purchase',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotation
ax.text(0.98, 0.95,
        f'IDE = {real["ide_pct_of_te"]:.1f}% (Disparate Treatment)\n'
        f'IIE = {real["iie_pct_of_te"]:.1f}% (Disparate Impact)\n'
        f'E-value = {real["e_value_point"]:.2f}',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{outdir}/figure6_real_hmda_decomposition.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/figure6_real_hmda_decomposition.pdf', bbox_inches='tight')
print("Saved figure6_real_hmda_decomposition.png/pdf")

print("\nAll validation figures generated successfully!")
