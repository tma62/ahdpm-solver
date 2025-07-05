import streamlit as st
import numpy as np
import sympy as sp
from scipy.special import erf as erf_np
import matplotlib.pyplot as plt

x = sp.Symbol('x')
lam = sp.Symbol('lambda')

def adm_bratu(lmbd_val=1, terms=3):
    u_terms = [sp.Integer(0)]
    A_terms = [sp.exp(u_terms[0])]
    for n in range(1, terms):
        u_sum = sum(u_terms[:n])
        An = sp.diff(sp.exp(u_sum), x, 0).subs(x, x)
        A_terms.append(An)
        s = sp.Symbol('s')
        An_s = A_terms[n].subs(x, s)
        integral = sp.integrate((x - s) * An_s, (s, 0, x))
        un = -lmbd_val * integral
        u_terms.append(un)
    return sp.simplify(sum(u_terms))

def numeric_adm_bratu(lmbd_val=1, terms=3):
    u_expr = adm_bratu(lmbd_val, terms)
    u_func = sp.lambdify(x, u_expr.subs(lam, lmbd_val), modules=['numpy', {'erf': erf_np}])
    return u_func

st.title("AHDPM Symbolic Solver (Bratu Equation Only)")

terms = st.slider("Number of symbolic terms", 1, 5, 3)
lambda_val = st.slider("Lambda (λ)", 0.1, 5.0, 1.0, step=0.1)

if st.button("Solve"):
    st.write(f"Solving Bratu equation with λ = {lambda_val} and {terms} terms...")
    adm_func = numeric_adm_bratu(lambda_val, terms)
    x_vals = np.linspace(0.01, 1, 200)
    try:
        u_vals = adm_func(x_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, u_vals, label="ADM", color="blue")
        ax.set_title("ADM Approximation for Bratu Equation")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Evaluation error: {e}")
