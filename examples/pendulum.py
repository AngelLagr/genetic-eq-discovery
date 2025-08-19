# Exemple d'utilisation du module pendulum_animation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split
from src.dataset_generator import generate_time_dataset
from src.geq_discovery import load_dataset, train_symbolic_regressor
from src.utils import convert_formula

def create_pendulum_gif(csv_file, gif_name):
    df = pd.read_csv(csv_file)
    time = df[df.columns[0]].values
    angle = df[df.columns[1]].values
    L = 1.0
    x = L * np.sin(angle)
    y = -L * np.cos(angle)
    fig, ax = plt.subplots()
    ax.set_xlim(-L-0.2, L+0.2)
    ax.set_ylim(-L-0.2, L+0.2)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    # Ajout du titre selon le fichier
    if 'generated' in gif_name:
        ax.set_title("Pendulum - Generated Data")
    else:
        ax.set_title("Pendulum - Real Data")
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    def update(frame):
        line.set_data([0, x[frame]], [0, y[frame]])
        time_text.set_text(f'Time = {time[frame]:.2f} s, Angle = {angle[frame]:.2f} rad')
        return line, time_text
    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=50)
    ani.save(gif_name, writer=PillowWriter(fps=20))
    plt.close(fig)

# Physics formula for a pendulum
formula = "np.cos(2 * np.pi * x1 / (2 * np.pi * np.sqrt(1 / 9.81))) * np.exp(-0.25 * x1)"
feature_names = ["x1"]

df = generate_time_dataset(formula, feature_names, n_samples=200)
df.to_csv("datasets/pendulum_dataset.csv", index=False)
X, y = load_dataset("datasets/pendulum_dataset.csv", target_col='y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_symbolic_regressor(X_train, y_train, X_test, y_test)


gen_formula = convert_formula(str(model._program))
gen_df = generate_time_dataset(gen_formula, feature_names, n_samples=200)
gen_df.to_csv("datasets/pendulum_dataset_generated.csv", index=False)

create_pendulum_gif("datasets/pendulum_dataset.csv", "output/pendulum_dataset.gif")
create_pendulum_gif("datasets/pendulum_dataset_generated.csv", "output/pendulum_dataset_generated.gif")
