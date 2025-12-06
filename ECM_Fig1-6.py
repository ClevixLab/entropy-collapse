import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10
})

# ===== FIGURE 1: Fixed Entropy Dynamics =====
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
time = np.arange(0, 51, 1)

# Healthy regime
H_healthy = 0.5 * np.exp(-0.02 * time) + 0.3 + 0.05 * np.random.randn(len(time))
H_healthy = np.clip(H_healthy, 0.1, 0.7)
ax1.plot(time, H_healthy, label='Healthy regime', linewidth=2.5, color='#2E86AB')
ax1.fill_between(time, H_healthy-0.03, H_healthy+0.03, alpha=0.2, color='#2E86AB')

# Collapse regime
H_collapse = 0.55 * np.exp(-0.12 * time) + 0.02 + 0.01 * np.random.randn(len(time))
H_collapse = np.clip(H_collapse, 0, 0.6)
ax1.plot(time, H_collapse, label='Collapse regime', linewidth=2.5, color='#E94F37')
ax1.fill_between(time, H_collapse-0.02, H_collapse+0.02, alpha=0.2, color='#E94F37')

ax1.set_xlabel('Time step')
ax1.set_ylabel('Entropy H')
ax1.set_title('Entropy Dynamics in RL Agents', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_entropy_dynamics_fixed.png', bbox_inches='tight', transparent=True)

# ===== FIGURE 2: Fixed Heatmaps =====
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 4))

# Pre-collapse
np.random.seed(42)
pre_data = np.random.uniform(0.3, 0.8, size=(10, 5))
im1 = ax2a.imshow(pre_data, cmap='viridis', aspect='auto', vmin=0, vmax=0.8)
ax2a.set_title('Pre-collapse (High diversity)', fontweight='bold')
ax2a.set_xlabel('State')
ax2a.set_ylabel('Agent')
cbar1 = plt.colorbar(im1, ax=ax2a, shrink=0.8)
cbar1.set_label('Entropy')

# Post-collapse với scale đúng
post_data = np.random.uniform(0.05, 0.1, size=(10, 5))
im2 = ax2b.imshow(post_data, cmap='viridis', aspect='auto', vmin=0, vmax=0.12)
ax2b.set_title('Post-collapse (Entropy ↓ → Homogenization)', fontweight='bold')
ax2b.set_xlabel('State')
ax2b.set_ylabel('Agent')
cbar2 = plt.colorbar(im2, ax=ax2b, shrink=0.8)
cbar2.set_label('Entropy')

plt.tight_layout()
plt.savefig('02_entropy_heatmap_fixed.png', bbox_inches='tight', transparent=True)

# ===== FIGURE 3: Fixed Infection Curve =====
fig3, ax3 = plt.subplots(figsize=(7, 4.5))
waves = np.arange(0, 11, 1)
fraction = 1 / (1 + np.exp(-1.5 * (waves - 5)))
ax3.plot(waves, fraction, 'o-', linewidth=3, markersize=8, 
         label='Fraction collapsed', color='#8A4F7D')

# Clean annotations
ax3.annotate('Slow initial spread', xy=(2, 0.1), xytext=(0.5, 0.25),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)
ax3.annotate('Rapid contagion', xy=(5, 0.5), xytext=(6, 0.7),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)
ax3.annotate('Saturation', xy=(8, 0.9), xytext=(7, 0.6),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)

ax3.set_xlabel('Infection wave')
ax3.set_ylabel('Fraction collapsed')
ax3.set_title('Viral Propagation of Collapse', fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03_infection_curve_fixed.png', bbox_inches='tight', transparent=True)

# ===== FIGURE 4: Fixed ECM Overview (SIMPLIFIED 2D) =====
fig4, ax4 = plt.subplots(figsize=(8, 5))
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = 2 * np.exp(-0.5*(X**2 + Y**2)) - 0.5*np.exp(-0.3*((X-2)**2 + (Y-2)**2))

# Chỉ contourf, không contour labels chồng
contour = ax4.contourf(X, Y, Z, levels=15, cmap='RdYlBu_r', alpha=0.8)

# Regions với text đúng
ax4.text(0, 0, 'High-entropy\nexploratory regime', fontsize=12, 
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
ax4.text(2.2, 2.2, 'Low-entropy\ncollapse basin', fontsize=12,
         ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

# Arrow
ax4.annotate('', xy=(1.8, 1.8), xytext=(0.5, 0.5),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='black', shrinkA=5, shrinkB=5))
ax4.text(1.0, 1.0, 'Entropy drift\nvia coupling & imitation', 
         fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

ax4.set_xlabel('Behavioural dimension 1')
ax4.set_ylabel('Behavioural dimension 2')
ax4.set_title('ECM Conceptual Overview: Entropy Landscape', fontweight='bold')
plt.colorbar(contour, ax=ax4, label='Entropy level')
plt.tight_layout()
plt.savefig('04_ecm_overview_fixed.png', bbox_inches='tight', transparent=True)

# ===== FIGURE 5: LLM Trace (SIMPLIFIED) =====
fig5, ax5 = plt.subplots(figsize=(9, 5))
np.random.seed(123)
messages = np.arange(0, 20, 1)
tokens = np.arange(0, 20, 1)
M, T = np.meshgrid(messages, tokens)
Z_entropy = 0.8 * np.exp(-0.15 * M) * (0.7 + 0.3 * np.exp(-0.1 * T))
Z_entropy = np.clip(Z_entropy, 0.1, 0.9)

heatmap = ax5.pcolormesh(T, M, Z_entropy.T, cmap='plasma', shading='auto')
ax5.set_xlabel('Token position')
ax5.set_ylabel('Message index')
ax5.set_title('LLM-Agent Conversation Entropy Trace', fontweight='bold')

plt.colorbar(heatmap, ax=ax5, label='Token-level entropy')
plt.tight_layout()
plt.savefig('05_llm_entropy_trace_fixed.png', bbox_inches='tight', transparent=True)

# ===== FIGURE 6: Fixed ERP Recovery =====
fig6, ax6 = plt.subplots(figsize=(7, 4.5))
time_erp = np.arange(0, 61, 1)

no_erp = 0.6 * np.exp(-0.1 * time_erp) + 0.02
weak_erp = 0.6 * np.exp(-0.07 * time_erp) + 0.15 * (1 - np.exp(-0.05 * (time_erp-20))) * (time_erp>20)
strong_erp = 0.6 * np.exp(-0.05 * time_erp) + 0.3 * (1 - np.exp(-0.1 * (time_erp-15))) * (time_erp>15)
strong_erp = np.clip(strong_erp, 0.25, 0.65)

ax6.plot(time_erp, no_erp, 's-', label='No ERP', linewidth=2.5, markersize=5, color='#E94F37')
ax6.plot(time_erp, weak_erp, 'o-', label='Weak ERP', linewidth=2.5, markersize=6, color='#F6AA1C')
ax6.plot(time_erp, strong_erp, 'D-', label='Strong ERP', linewidth=2.5, markersize=6, color='#2E86AB')

ax6.set_xlabel('Time step')
ax6.set_ylabel('Entropy H')
ax6.set_title('Entropy Reinjection Protocol (ERP): Collapse vs Recovery', fontweight='bold')
ax6.legend(loc='upper right')
ax6.grid(True, alpha=0.3)

# Clean annotations
ax6.annotate('Collapse phase', xy=(25, 0.15), xytext=(30, 0.25),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)
ax6.annotate('Recovery with\nStrong ERP', xy=(45, 0.5), xytext=(50, 0.6),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)

plt.tight_layout()
plt.savefig('06_erp_recovery_fixed.png', bbox_inches='tight', transparent=True)

plt.show()