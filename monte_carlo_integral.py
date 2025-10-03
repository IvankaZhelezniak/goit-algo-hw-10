"""

Завдання:
- Обчислити ∫_a^b f(x) dx методом Монте-Карло.
- Перевірити результат аналітично та за допомогою SciPy: scipy.integrate.quad.

Тут свідомо беру f(x) = x^2 на [0, 2], бо для неї є точна формула.
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np
import scipy.integrate as spi


# 1) ФУНКЦІЯ, ЯКУ ІНТЕГРУЮ

def f_vec(x: np.ndarray) -> np.ndarray:
    """
    Векторизована версія функції: f(x) = x^2 для масиву x.
    """
    return x**2

def f_scalar(x: float) -> float:
    """
    Скалярна версія тієї ж функції: f(x) = x^2 для float.
    Саме таку сигнатуру очікує scipy.integrate.quad.
    """
    return x * x

# 2) АНАЛІТИЧНИЙ ВІДПОВІДНИК ДЛЯ ПЕРЕВІРКИ
def analytic_integral_f(a: float, b: float) -> float:
    """
    ∫_a^b x^2 dx = (b^3 - a^3) / 3
    """
    return (b**3 - a**3) / 3.0

# 3) МЕТОД МОНТЕ-КАРЛО
def monte_carlo_integral(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    seed: int | None = 42,
) -> Dict[str, float]:
    """
    Оцінка I = ∫_a^b f(x) dx методом Монте-Карло:

      1) U_i ~ Uniform(a, b), i=1..N
      2) E[f(U)] ≈ (1/N) Σ f(U_i)
      3) I ≈ (b-a) * E[f(U)]

    Повертає оцінку, стандартну похибку, 95% довірчий інтервал.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)          # рівномірні точки на [a, b]
    fx = f(x)                              # значення f у цих точках (векторно)

    mean_fx = fx.mean()                    # оцінка E[f(U)]
    var_fx = fx.var(ddof=1) if n > 1 else 0.0
    estimate = (b - a) * mean_fx
    se = (b - a) * np.sqrt(var_fx / n)     # SE = (b-a)*sqrt(Var[f(U)]/N)
    ci_low = estimate - 1.96 * se          # 95% ДІ
    ci_high = estimate + 1.96 * se

    return {
        "estimate": float(estimate),
        "std_error": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }

# 4) ПЕРЕВІРКА ЧЕРЕЗ SciPy (quad) + АНАЛІТИКА

def check_with_quad_and_analytic(
    a: float, b: float, n_for_mc: int = 1_000_000, seed: int = 42
) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
    """
    Повертає:
      - res_mc: результат Монте-Карло (оцінка, SE, 95% ДІ)
      - val_analytic: аналітичне значення
      - (quad_val, quad_err): результат quad та його оцінка похибки
    """
    # Монте-Карло (векторизована f)
    res_mc = monte_carlo_integral(f_vec, a, b, n=n_for_mc, seed=seed)

    # Аналітика
    val_analytic = analytic_integral_f(a, b)

    # SciPy quad — використовую скалярну версію функції
    quad_val, quad_err = spi.quad(f_scalar, a, b)

    return res_mc, val_analytic, (quad_val, quad_err)


# 5) ДЕМО / ВИВІД
def run_demo() -> None:
    a, b = 0.0, 2.0

    # Показую збіжність Монте-Карло при зростанні N
    true_value = analytic_integral_f(a, b)
    print(f"Аналітичне значення: {true_value:.12f}  (очікувано 8/3 ≈ 2.6666666667)\n")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        res = monte_carlo_integral(f_vec, a, b, n=n, seed=42)
        print(
            f"n={n:>9,} | "
            f"MC={res['estimate']:.10f}  "
            f"SE≈{res['std_error']:.6f}  "
            f"95% CI=({res['ci_low']:.10f}, {res['ci_high']:.10f})"
        )

    # Перевірка: SciPy quad + зведене порівняння
    print("\nПеревірка через SciPy.quad:")
    quad_val, quad_err = spi.quad(f_scalar, a, b)
    print(f"  quad result = {quad_val:.12f},   reported abs error ≈ {quad_err:.2e}")

    # Зведене порівняння на великому N
    n_big = 1_000_000
    mc, analytic_val, (qv, qe) = check_with_quad_and_analytic(a, b, n_for_mc=n_big, seed=42)
    print("\nЗведене порівняння (N = {:,}):".format(n_big))
    print(f"  MC estimate        = {mc['estimate']:.12f}")
    print(f"  MC 95% CI          = ({mc['ci_low']:.12f}, {mc['ci_high']:.12f})")
    print(f"  Аналітичне значення= {analytic_val:.12f}")
    print(f"  SciPy quad         = {qv:.12f} (±{qe:.2e})")


if __name__ == "__main__":
    run_demo()
