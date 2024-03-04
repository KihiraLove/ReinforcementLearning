import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gridworld import GridWorld

world = \
    """
    wwwwwwwwwwwwwwwww
    wa   o   w     gw
    w               w
    www  o   www  www
    w               w
    wwwww    o    www
    w     ww        w
    wwwwwwwwwwwwwwwww
    """

np.random.seed(42)
env = GridWorld(world, slip=0.2, log=False, max_episode_step=1000)
gamma = 0.9

# Policy Iteration
V = np.zeros((env.state_count, 1))
pi = np.random.choice(env.action_values, size=env.state_count)  # random policy
pi_prev = np.random.choice(env.action_values, size=env.state_count)

i = 0
v_values = []

while np.sum(np.abs(pi - pi_prev)) > 0:  # until no policy change
    pi_prev = pi.copy()
    P_ss = np.squeeze(np.take_along_axis(env.P_sas, pi.reshape(-1, 1, 1), axis=1), axis=1)
    R_s = np.take_along_axis(env.R_sa, pi.reshape(-1, 1), axis=1)
    V = R_s + gamma * np.matmul(P_ss, V)
    pi = np.argmax(env.R_sa + gamma * np.squeeze(np.matmul(env.P_sas, V)), axis=1)
    v_values.append(np.max(np.abs(V)))
    i += 1

report = f"Converged in {i} iterations\n"
report += f"Pi_*= {pi}\n"
report += f"V_*= {V.flatten()}\n"
print(report)

plt.plot(v_values, lw=3, ls='--')
plt.ylabel('$|V|_{\infty}$', fontsize=16)
plt.xticks(range(len(v_values)), labels=["$\pi_{" + f"{e}" + "}$" for e in range(len(v_values))])
plt.xlabel('Policy', fontsize=16)
plt.tight_layout()
plt.savefig("fig.png")