import numpy as np
import matplotlib.pyplot as plt

# Definieer matrices A en B
A = np.array([[-1, -4], [0, -3]])  # Payoff matrix voor speler 1 Prisonners dilemma
B = np.array([[-1, 0], [-4, -3]])  # Payoff matrix voor speler 2 Prisonners dilemma

# Definieer matrices A en B
#A = np.array([[12, 0], [11, 10]])  # Payoff matrix voor speler 1 Subsidy Game
#B = np.array([[12, 11], [0, 10]])  # Payoff matrix voor speler 2 Subsidy Game

# Definieer matrices A en B
#A = np.array([[3, 0], [0, 2]])  # Payoff matrix voor speler 1 battle of the sexes
#B = np.array([[2, 0], [0, 3]])  # Payoff matrix voor speler 2 battle of the sexes



# Definieer de differentiaalvergelijkingen voor x1 en y1
def dx1_dt(x, y):
    Ay = np.dot(A, y)
    return x[0] * (Ay[0] - np.dot(x, Ay))

def dy1_dt(x, y):
    Bx = np.dot(x, B)
    return y[0] * (Bx[0] - np.dot(x, np.dot(B, y)))


# Define ε-greedy and Lenient Boltzmann Q-learning functions
def epsilon_greedy(Q, epsilon):
    print(Q)
    if np.random.rand() < epsilon:
        print('ok')
        return np.random.randint(2)  # random action
    else:
        return np.argmax(Q)  # exploit current knowledge

def lenient_boltzmann(Q, tau):
    probs = np.exp(Q / tau) / np.sum(np.exp(Q / tau))
    return np.random.choice(2, p=probs) # dimension 2

def learn(Q, start_point, alpha, beta, epsilon, tau, iterations=1000):
    traj = [start_point]
    for _ in range(iterations):
        x, y = traj[-1]
        action = epsilon_greedy(Q,epsilon) # lenient_boltzmann(Q, tau) # swap to other trajectory
        if action == 0:  # player 1 cooperates
            x_next = x + alpha * dx1_dt([x, 1 - x], [y, 1 - y])
            y_next = y + beta * dy1_dt([x, 1 - x], [y, 1 - y])
        else:  # player 1 defects
            x_next = x + alpha * dx1_dt([1 - x, x], [1 - y, y])
            y_next = y + beta * dy1_dt([1 - x, x], [1 - y, y])
        traj.append((x_next, y_next))

        # Calculate rewards
        reward = np.dot(np.dot([x_next, 1 - x_next], A), [y_next, 1 - y_next]) if action == 0 else np.dot(np.dot([1 - x_next, x_next], B), [1 - y_next, y_next])

        # Update Q-values
        Q[action] += alpha * (reward - Q[action])

        if np.allclose(traj[-1], (0, 0), atol=1e-3):  # convergence condition
            break
    return traj

# Maak een rooster van punten in het eenheidsvierkant
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y)

# Bereken de differentiaalvergelijkingen op het rooster
DX1 = np.zeros_like(X)
DY1 = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        DX1[i, j] = dx1_dt([X[i, j], 1 - X[i, j]], [Y[i, j], 1 - Y[i, j]])
        DY1[i, j] = dy1_dt([X[i, j], 1 - X[i, j]], [Y[i, j], 1 - Y[i, j]])

# Plot de vectorvelden
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, DX1, DY1, color='blue', scale=10)

# initialisation of points
trajectories = [(0.5, 0.5),(0.9, 0.2),(0.2, 0.9),(0.8, 0.9),(0.9, 0.8),(0.2,0.4),(0.6,0.3),(0.1,0.6)]

# Simulate and plot learning trajectories
for start_point in trajectories:
    traj = learn(Q=np.zeros(2),start_point = start_point, alpha=0.01, beta=0.01, epsilon=0.1, tau=0.1, iterations=1000)
    traj_x, traj_y = zip(*traj)
    plt.plot(traj_x, traj_y, color='red', linestyle='dashed', linewidth=2)

# battle of the sexes
plt.xlabel('player 1, probability of playing Cooperation') # change label when other matrix game is used
plt.ylabel('player 2, probability of playing Cooperation') # change label when other matrix game is used
plt.title('Directional filed plot: Prisoners Dilemma') # change label when other label game is used
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()
