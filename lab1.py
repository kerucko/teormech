import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def chetv(x, y):
    if x >= 0 and y >= 0:
        return 1
    if x <= 0 and y >= 0:
        return 2
    if x <= 0 and y <= 0:
        return 3
    if x >= 0 and y <= 0:
        return 4


def compare_tan(x0, y0, x1, y1, ch):
    if x1 == 0:
        t1 = 99999
    else:
        t1 = y1 / x1
    if x0 == 0:
        t0 = 99999
    else:
        t0 = y0 / x0
    if ch % 2:
        if t1 <= t0:
            return 1
        else:
            return -1
    else:
        if t1 <= t0:
            return -1
        else:
            return 1


def direction(x0, y0, x1, y1):
    ch0 = chetv(x0, y0)
    ch1 = chetv(x1, y1)
    if ch1 == ch0:
        return compare_tan(x0, y0, x1, y1, ch1)
    else:
        if ch0 == 1 and ch1 == 2 or ch0 == 2 and ch1 == 3 or ch0 == 3 and ch1 == 4 or ch0 == 4 and ch1 == 1:
            return -1
        elif (ch1 % 2) == (ch0 % 2):
            return -compare_tan(x0, y0, x1, y1, ch1)
        else:
            return 1


def Rotation(X, Y, alpha):
    RX = X * np.cos(alpha) - Y * np.sin(alpha)
    RY = X * np.sin(alpha) + Y * np.cos(alpha)
    return RX, RY


def animation(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    ALine.set_data([X[i], X[i] + AX[i]], [Y[i], Y[i] + AY[i]])
    d = direction(X[i - 1], Y[i - 1], X[i], Y[i])
    CVector.set_data([X[i], X[i] + ANX[i] * CV[i] * d], [Y[i], Y[i] + ANY[i] * CV[i] * d])
    print(d)

    RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowVX + X[i] + VX[i], RArrowVY + Y[i] + VY[i])
    RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AArrow.set_data(RArrowAX + X[i] + AX[i], RArrowAY + Y[i] + AY[i])
    RArrowCX, RArrowCY = Rotation(ArrowX, ArrowY, math.atan2(ANY[i], ANX[i]))
    CVArrow.set_data(RArrowCX + X[i] + ANX[i] * CV[i] * d, RArrowCY + Y[i] + ANY[i] * CV[i] * d)
    return P, VLine, ALine, VArrow, AArrow, CVector, CVArrow,


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
ax1.set(xlim=[-5, R], ylim=[-R, R])

ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')
VLine, = ax1.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r')
ALine, = ax1.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], 'y')
CVector, = ax1.plot([X[0], X[0] + ANX[0] * CV[0]], [Y[0], Y[0] + ANY[0] * CV[0]], 'orange')

ArrowX = np.array([-0.2 * S, 0, -0.2 * S])
ArrowY = np.array([0.1 * S, 0, -0.1 * S])
RArrowVX, RArrowVY = Rotation(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowVX + X[0] + VX[0], RArrowVY + Y[0] + VY[0], 'r')
RArrowAX, RArrowAY = Rotation(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AArrow, = ax1.plot(RArrowAX + X[0] + AX[0], RArrowAY + Y[0] + AY[0], 'y')
RArrowCX, RArrowCY = Rotation(ArrowX, ArrowY, math.atan2(ANY[0], ANX[0]))
CVArrow, = ax1.plot(RArrowCX + X[0] + ANX[0] * CV[0], RArrowCY + Y[0] + ANY[0] * CV[0], 'orange')

anim = FuncAnimation(fig, animation, frames=1500, interval=6, blit=True, repeat=True)

plt.show()
