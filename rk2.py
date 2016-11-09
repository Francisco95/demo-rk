#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementa la solucion de RK2 para el problema de un
pendulo real con largo 1.
"""

import numpy as np
import matplotlib.pyplot as plt


# constantes
g = 9.8

# condiciones iniciales
phi_t0 = np.pi / 2.
w_t0 = 0.

# Solucion para pequeñas oscilaciones
freq = np.sqrt(g)
t = np.linspace(0, 4 * 2 * np.pi / freq, 400)
phi_po = phi_t0 * np.cos(freq * t)

plt.figure(1)
plt.clf()

plt.plot(t, phi_po, label=u'pequeñas oscilaciones')


# Usando RK2

def f(phi, omega):
    output = [omega, -g * np.sin(phi)]
    return output

def calc_k1(f, phi_n, omega_n, h):
    fn = f(phi_n, omega_n)
    output = [h * fn[0], h * fn[1]]
    return output

def calc_k2(f, phi_n, omega_n, h):
    k1 = calc_k1(f, phi_n, omega_n, h)
    f_middle = f(phi_n + k1[0]/2, omega_n + k1[1]/2)
    output = [h * f_middle[0], h * f_middle[1]]
    return output

def rk2_step(f, phi_n, omega_n, h):
    k2 = calc_k2(f, phi_n, omega_n, h)
    phi_next = phi_n + k2[0]
    omega_next = omega_n + k2[1]
    output = [phi_next, omega_next]
    return output

Nsteps = 400
h = 4 * 2 * np.pi / freq / Nsteps

phi_arr = np.zeros(Nsteps)
omega_arr = np.zeros(Nsteps)

# Condiciones iniciales
phi_arr[0] = phi_t0
omega_arr[0] = w_t0

for i in range(1, Nsteps):
    phi_arr[i], omega_arr[i] = rk2_step(f, phi_arr[i-1], omega_arr[i-1], h)


plt.plot(t, phi_arr, label='rk2')



plt.legend()
plt.show()
