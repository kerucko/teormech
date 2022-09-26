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
    RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowVX + X[i] + VX[i], RArrowVY + Y[i] + VY[i])
    RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(RArrowAX + X[i] + AX[i], RArrowAY + Y[i] + AY[i])
    CVector.set_data([X[i], X[i] + ANX[i] * CV[i]], [Y[i], Y[i] + ANY[i] * CV[i]])
    return P, VLine, ALine, VArrow, AArrow, CVector,


t = sp.Symbol('t')
R = 75
S = 4

# r = sp.cos(6 * t)
# fi = t + 0.2 * sp.cos(3 * t)

# r = 2 + sp.sin(6 * t)
# fi = 6.5 * t + 1.2 * sp.cos(6 * t)

# x = r * sp.cos(fi)
# y = r * sp.sin(fi)

x = t ** 2
y = 10 * sp.sin(t)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
V = sp.sqrt(Vx ** 2 + Vy ** 2)
Afull = sp.sqrt(Ax ** 2 + Ay ** 2)
Atan = sp.diff(V)
An = sp.sqrt(Afull ** 2 - Atan ** 2)
Сurve = V ** 2 / An

T = np.linspace(0, 8, 1500)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
ANX = np.zeros_like(T)
ANY = np.zeros_like(T)
CV = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    ANX[i] = VY[i] / math.sqrt(VX[i] ** 2 + VY[i] ** 2)
    ANY[i] = -VX[i] / math.sqrt(VX[i] ** 2 + VY[i] ** 2)
    CV[i] = sp.Subs(Сurve, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[0, R], ylim=[-R, R])

ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
ALine, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'y')
ArrowX = np.array([-0.2 * S, 0, -0.2 * S])
ArrowY = np.array([0.1 * S, 0, -0.1 * S])
RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowVX + X[0] + VX[0], RArrowVY + Y[0] + VY[0], 'r')
RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AArrow, = ax1.plot(RArrowAX + X[0] + AX[0], RArrowAY + Y[0] + AY[0], 'y')
CVector, = ax1.plot([X[0], X[0] + ANX[0] * CV[0]], [Y[0], Y[0] + ANY[0] * CV[0]], 'orange')

anim = FuncAnimation(fig, animation, frames=1500, interval=6, blit=True, repeat=True)

plt.show()
