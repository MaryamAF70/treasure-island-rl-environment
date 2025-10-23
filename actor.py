import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

def draw_box(x, y, width, height, label, color, text_color='black'):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.02",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center',
            fontsize=12, color=text_color)

def draw_arrow(start, end, text=None, color='black', style='->', linewidth=2, text_offset=(0, 0), fontsize=11):
    arrow = FancyArrowPatch(start, end,
                            arrowstyle=style,
                            color=color,
                            linewidth=linewidth,
                            mutation_scale=15)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=fontsize, ha='center', va='center', color=color)

# Draw components
draw_box(1, 3.5, 2.5, 1.5, 'Actor\n(Policy π)', '#a3c9f1', text_color='navy')
draw_box(1, 1.0, 2.5, 1.5, 'Critic\n(Value V)', '#a1e6a1', text_color='darkgreen')
draw_box(6, 2.0, 2.8, 2.0, 'Environment', '#eaeaea')

# Draw arrows
draw_arrow((3.6, 4.25), (6.0, 3.8), 'Action a', text_offset=(0, 0.3))
draw_arrow((6.0, 2.2), (3.6, 1.75), 'Reward r, State s\'', text_offset=(0, -0.4))
draw_arrow((2.25, 3.5), (2.25, 2.5), 'State s', text_offset=(-0.8, 0))
draw_arrow((2.25, 2.0), (2.25, 3.0), 'TD Error δ', color='red', style='-|>', text_offset=(1.0, 0))

plt.tight_layout()
plt.show()
