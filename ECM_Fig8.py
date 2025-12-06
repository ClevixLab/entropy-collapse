import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Set consistent style với 6 figures trước
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

def create_phase_diagram():
    """Create professional phase diagram for ECM"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Parameters
    IE = np.linspace(0, 1.0, 500)  # Innovation entropy
    CB_plus_D = 0.4  # Critical threshold (coupling + drift)
    
    # Equilibrium entropy: H* = max(0, IE - (CB+D))
    H_eq = np.maximum(0, IE - CB_plus_D)
    
    # Plot critical boundary
    ax.plot(IE, H_eq, 'k-', linewidth=3, label='Critical boundary $IE = CB + D$', zorder=10)
    
    # Fill healthy regime (IE > CB+D)
    healthy_mask = IE >= CB_plus_D
    ax.fill_between(IE[healthy_mask], H_eq[healthy_mask], 1.0,
                    color='#2E86AB', alpha=0.25, label='Healthy regime\n(entropy sustained)')
    
    # Fill collapse basin (IE < CB+D)
    collapse_mask = IE < CB_plus_D
    ax.fill_between(IE[collapse_mask], 0, H_eq[collapse_mask],
                    color='#E94F37', alpha=0.25, label='Collapse basin\n(entropy lost)')
    
    # Add critical point marker
    ax.plot(CB_plus_D, 0, 'ko', markersize=10, markerfacecolor='white', 
            markeredgewidth=2, label='Critical point $(IE^*, 0)$')
    
    # Annotations với arrows
    # Arrow từ collapse sang healthy
    ax.annotate('Entropy recovery\nrequires $\\uparrow IE$ or $\\downarrow (CB+D)$',
                xy=(0.6, 0.7), xytext=(0.25, 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', 
                                connectionstyle="arc3,rad=0.2"),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Monomorphic attractor annotation
    ax.annotate('Monomorphic\nattractor $H^*=0$',
                xy=(0.2, 0.05), xytext=(0.05, 0.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'),
                fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Labels và titles
    ax.set_xlabel('Innovation entropy $IE$', fontsize=12)
    ax.set_ylabel('Equilibrium entropy $H^*$', fontsize=12)
    ax.set_title('Phase Diagram of ECM: Healthy vs Collapse Regimes', 
                 fontsize=14, fontweight='bold')
    
    # Grid và limits
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend với vị trí tốt
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Thêm inset với ECM equation
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right')
    ax_inset.text(0.5, 0.5, 
                  r'$H_{t+1} = H_t + IE_t - CB_t - D_t$'
                  '\n\nCollapse when:'
                  '\n$IE_t < CB_t + D_t$',
                  ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    ax_inset.axis('off')
    
    # Thêm shaded region cho transition
    ax.axvspan(CB_plus_D-0.05, CB_plus_D+0.05, alpha=0.15, color='gray', 
               label='Phase transition\nregion')
    
    plt.tight_layout()
    plt.savefig('08_phase_diagram_pro.png', bbox_inches='tight', transparent=True, dpi=600)
    plt.savefig('08_phase_diagram_pro.pdf', bbox_inches='tight', transparent=True)  # Vector format
    plt.close()
    
    print("✅ Phase diagram created: '08_phase_diagram_pro.png' (PNG)")
    print("✅ Phase diagram created: '08_phase_diagram_pro.pdf' (PDF)")

if __name__ == "__main__":
    create_phase_diagram()