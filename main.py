import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def Rotation(X, Y, alpha):
    RX = X * np.cos(alpha) - Y * np.sin(alpha)
    RY = X * np.sin(alpha) + Y * np.cos(alpha)
    return RX, RY


def animation(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    ALine.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])

    RArrowX, RArrowY = Rotation(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX + X[i] + VX[i], RArrowY + Y[i] + VY[i])

    Circle.set_data(XC[i] + R * np.cos(Phi), YC + R * np.sin(Phi))
    return P, VLine, ALine, VArrow, Circle,


R = 4
Omega = 1
t = sp.Symbol('t')

x = R * (Omega * t - sp.sin(Omega * t))
y = R * (1 - sp.cos(Omega * t))
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
xC = R * Omega * t
yC = R

T = np.linspace(0, 10, 1000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
XC = np.zeros_like(T)
YC = R

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    XC[i] = sp.Subs(xC, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-R, 12 * R], ylim=[-R, 3 * R])

ax1.plot(X, Y)
ax1.plot([X.min(), X.max()], [0, 0], 'black')

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
ALine, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'y')

Phi = np.linspace(0, 6.28, 100)
Circle, = ax1.plot(XC[0] + R * np.cos(Phi), YC + R * np.sin(Phi), 'g')

ArrowX = np.array([-0.2 * R, 0, -0.2 * R])
ArrowY = np.array([0.1 * R, 0, -0.1 * R])
RArrowX, RArrowY = Rotation(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX + X[0] + VX[0], RArrowY + Y[0] + VY[0])

anim = FuncAnimation(fig, animation, frames=1000, interval=2, blit=True, repeat=True)

plt.show()
