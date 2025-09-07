#!/usr/bin/env python3
# math_toolbox.py — Pure-stdlib math CLI (no external dependencies)
# Features:
# - Safe expression evaluator (AST-based) with math functions
# - Solve: linear, quadratic; linear system (Gaussian elimination)
# - Numeric calculus: derivative (central difference), integral (Simpson/trapezoid)
# - Root finding: bisection or Newton
# - Matrix ops: det, inverse (Gauss-Jordan), multiply
# Usage examples at bottom comment

import argparse, ast, math, operator, sys
from typing import Any, Dict, List, Tuple
#

# ---- Safe expression evaluator --------
_ALLOWD_FUNCS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan}
_ALLOWED_NAMES = {**_ALLOWED_FUNCS, **_ALLOWED_CONSTS}

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow 
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def safe_eval(expr: str, variables: Dict[str, float] = None) -> float:
    """
    Evaluate a math expression safely with AST.
    Supports numbers, variables, + - * / // % **, parentheses, and math.* functions.
    variables: dict of allowed variable names -> values
    """
    variables = variables or {}
    node = ast.parse(expr.replace("^", "**"), mode="eval")

    def _eval(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Num):  # py<=3.7
            return n.n
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Only numeric constants are allowed.")
        if isinstance(n, ast.BinOp):
            if type(n.op) not in _BIN_OPS:
                raise ValueError("Operator not allowed.")
            return _BIN_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in _UNARY_OPS:
                raise ValueError("Unary operator not allowed.")
            return _UNARY_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Name):
            if n.id in variables:
                return variables[n.id]
            if n.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[n.id]
            raise NameError(f"Name '{n.id}' is not allowed.")
        if isinstance(n, ast.Call):
            func = _eval(n.func)
            args = [_eval(a) for a in n.args]
            if not callable(func):
                raise ValueError("Attempted to call non-callable.")
            return func(*args)
        if isinstance(n, ast.Attribute):
            # allow math.sin style via allowed names
            if isinstance(n.value, ast.Name) and n.value.id == "math":
                attr = n.attr
                if attr in _ALLOWED_FUNCS:
                    return _ALLOWED_FUNCS[attr]
            raise ValueError("Attributes are not allowed.")
        if isinstance(n, ast.Tuple):
            return tuple(_eval(elt) for elt in n.elts)
        raise ValueError("Unsupported expression element.")
    return float(_eval(node))
# -------------------------------------------------------

# ------------------------- Algebra helpers ---------------------------
def solve_linear(a: float, b: float) -> Tuple[str, List[float]]:
    if a == 0:
        if b == 0:
            return "Infinite solutions", []
        return "No solution", []
    return "One solution", [-b / a]

def solve_quadratic(a: float, b: float, c: float) -> Tuple[str, List[float]]:
    if a == 0:
        return solve_linear(b, c)
    d = b*b - 4*a*c
    if d > 0:
        r = math.sqrt(d)
        return "Two real solutions", [(-b + r)/(2*a), (-b - r)/(2*a)]
    elif d == 0:
        return "One real solution", [-b/(2*a)]
    else:
        r = math.sqrt(-d)
        real = -b/(2*a)
        imag = r/(2*a)
        return "Two complex solutions", [complex(real, imag), complex(real, -imag)]

def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    # Solve Ax=b for square A (n x n)
    n = len(A)
    M = [row[:] + [bv] for row, bv in zip(A, b)]  # augmented

    for col in range(n):
        # pivot
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
        M[col], M[pivot] = M[pivot], M[col]
        # normalize pivot row
        fac = M[col][col]
        for j in range(col, n+1):
            M[col][j] /= fac
        # eliminate below
        for i in range(col+1, n):
            fac = M[i][col]
            for j in range(col, n+1):
                M[i][j] -= fac * M[col][j]
    # back substitution
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j]*x[j] for j in range(i+1, n))
    return x
# ---------------------------------------------------------------------

# ---------------------- Numeric calculus -----------------------------
def numeric_derivative(func_expr: str, x0: float, h: float = 1e-5) -> float:
    # 5-point central difference
    f = lambda x: safe_eval(func_expr, {"x": x})
    return ( -f(x0+2*h) + 8*f(x0+h) - 8*f(x0-h) + f(x0-2*h) ) / (12*h)

def numeric_integral(func_expr: str, a: float, b: float, n: int = 1000) -> float:
    # Simpson if n even else trapezoid
    if n <= 1:
        n = 2
    f = lambda x: safe_eval(func_expr, {"x": x})
    if n % 2 == 1:  # trapezoid
        h = (b-a)/n
        s = 0.5*(f(a)+f(b))
        for i in range(1, n):
            s += f(a + i*h)
        return s*h
    # Simpson's rule
    h = (b-a)/n
    s = f(a) + f(b)
    s += 4*sum(f(a + (2*i-1)*h) for i in range(1, n//2 + 0))
    s += 2*sum(f(a + 2*i*h) for i in range(1, n//2))
    return s*h/3.0

def bisection(func_expr: str, left: float, right: float, tol: float = 1e-8, maxit: int = 10_000) -> float:
    f = lambda x: safe_eval(func_expr, {"x": x})
    fl, fr = f(left), f(right)
    if fl == 0: return left
    if fr == 0: return right
    if fl*fr > 0:
        raise ValueError("Function must have opposite signs at the interval endpoints.")
    for _ in range(maxit):
        mid = (left+right)/2
        fm = f(mid)
        if abs(fm) < tol or (right-left)/2 < tol:
            return mid
        if fl*fm < 0:
            right, fr = mid, fm
        else:
            left, fl = mid, fm
    return (left+right)/2

def newton(func_expr: str, x0: float, tol: float = 1e-10, maxit: int = 10_000) -> float:
    f = lambda x: safe_eval(func_expr, {"x": x})
    df = lambda x: numeric_derivative(func_expr, x)
    x = x0
    for _ in range(maxit):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ValueError("Derivative is near zero; Newton may diverge.")
        step = fx/dfx
        x_new = x - step
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x
# ---------------------------------------------------------------------

# ------------------------ Matrix utilities ---------------------------
def mat_det(A: List[List[float]]) -> float:
    # LU-like via elimination (no pivot sign handling for brevity; use partial pivot)
    n = len(A)
    M = [row[:] for row in A]
    sign = 1.0
    det = 1.0
    for i in range(n):
        # pivot
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-12:
            return 0.0
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
            sign *= -1
        piv = M[i][i]
        det *= piv
        for j in range(i+1, n):
            factor = M[j][i] / piv
            for k in range(i, n):
                M[j][k] -= factor * M[i][k]
    return det * sign

def mat_inv(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    M = [row[:] + [1.0 if i==j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    # Gauss-Jordan
    for i in range(n):
        # pivot
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-12:
            raise ValueError("Matrix is singular.")
        M[i], M[pivot] = M[pivot], M[i]
        piv = M[i][i]
        for j in range(2*n):
            M[i][j] /= piv
        for r in range(n):
            if r == i: continue
            fac = M[r][i]
            for j in range(2*n):
                M[r][j] -= fac * M[i][j]
    return [row[n:] for row in M]

def mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m, p = len(A), len(A[0]), len(B[0])
    assert len(B) == m, "Incompatible shapes"
    return [[sum(A[i][k]*B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
# ---------------------------------------------------------------------

# --------------------------- CLI commands ----------------------------
def cmd_eval(args):
    print(safe_eval(args.expr))

def cmd_linear(args):
    status, roots = solve_linear(args.a, args.b)
    print(status)
    if roots:
        print("x =", roots[0])

def cmd_quadratic(args):
    status, roots = solve_quadratic(args.a, args.b, args.c)
    print(status)
    for i, r in enumerate(roots, 1):
        print(f"x{i} = {r}")

def cmd_system(args):
    # parse A and b from strings like "1,2;3,4" and "5,6"
    A = [[float(x) for x in row.split(",")] for row in args.A.split(";")]
    b = [float(x) for x in args.b.split(",")]
    sol = gaussian_elimination(A, b)
    for i, v in enumerate(sol, 1):
        print(f"x{i} = {v}")

def cmd_diff(args):
    print(numeric_derivative(args.func, args.x, args.h))

def cmd_int(args):
    print(numeric_integral(args.func, args.a, args.b, args.n))

def cmd_root(args):
    if args.method == "bisection":
        print(bisection(args.func, args.left, args.right, args.tol, args.maxit))
    else:
        print(newton(args.func, args.x0, args.tol, args.maxit))

def cmd_mdet(args):
    A = [[float(x) for x in row.split(",")] for row in args.A.split(";")]
    print(mat_det(A))

def cmd_minv(args):
    A = [[float(x) for x in row.split(",")] for row in args.A.split(";")]
    inv = mat_inv(A)
    for row in inv:
        print(", ".join(f"{v:.10g}" for v in row))

def cmd_mmul(args):
    A = [[float(x) for x in row.split(",")] for row in args.A.split(";")]
    B = [[float(x) for x in row.split(",")] for row in args.B.split(";")]
    C = mat_mul(A, B)
    for row in C:
        print(", ".join(f"{v:.10g}" for v in row))

def build_parser():
    p = argparse.ArgumentParser(description="Pure-stdlib Math Toolbox CLI (no installs).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("eval", help="Safely evaluate an expression, e.g. 'sin(pi/6)+cos(0)'")
    s.add_argument("expr"); s.set_defaults(func=cmd_eval)

    s = sub.add_parser("linear", help="Solve ax + b = 0")
    s.add_argument("a", type=float); s.add_argument("b", type=float)
    s.set_defaults(func=cmd_linear)

    s = sub.add_parser("quadratic", help="Solve ax^2 + bx + c = 0")
    s.add_argument("a", type=float); s.add_argument("b", type=float); s.add_argument("c", type=float)
    s.set_defaults(func=cmd_quadratic)

    s = sub.add_parser("system", help='Solve linear system Ax=b. A like "1,2;3,4"  b like "5,6"')
    s.add_argument("--A", required=True); s.add_argument("--b", required=True)
    s.set_defaults(func=cmd_system)

    s = sub.add_parser("derivative", help="Numeric derivative at x (5-point central)")
    s.add_argument("--func", required=True); s.add_argument("--x", type=float, required=True)
    s.add_argument("--h", type=float, default=1e-5)
    s.set_defaults(func=cmd_diff)

    s = sub.add_parser("integral", help="Numeric integral on [a,b] (Simpson/trapezoid)")
    s.add_argument("--func", required=True); s.add_argument("--a", type=float, required=True); s.add_argument("--b", type=float, required=True)
    s.add_argument("--n", type=int, default=1000)
    s.set_defaults(func=cmd_int)

    s = sub.add_parser("root", help="Root finding by bisection or Newton")
    s.add_argument("--func", required=True)
    s.add_argument("--method", choices=["bisection", "newton"], default="bisection")
    s.add_argument("--left", type=float, help="Left bound for bisection")
    s.add_argument("--right", type=float, help="Right bound for bisection")
    s.add_argument("--x0", type=float, help="Initial guess for Newton")
    s.add_argument("--tol", type=float, default=1e-10)
    s.add_argument("--maxit", type=int, default=10000)
    s.set_defaults(func=cmd_root)

    s = sub.add_parser("mdet", help='Matrix determinant. A like "1,2;3,4"')
    s.add_argument("--A", required=True); s.set_defaults(func=cmd_mdet)

    s = sub.add_parser("minv", help='Matrix inverse. A like "1,2;3,4"')
    s.add_argument("--A", required=True); s.set_defaults(func=cmd_minv)

    s = sub.add_parser("mmul", help='Matrix multiply: A*B. A/B like "1,2;3,4"')
    s.add_argument("--A", required=True); s.add_argument("--B", required=True)
    s.set_defaults(func=cmd_mmul)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    # quick sanity for root subcommand
    if args.cmd == "root":
        if args.method == "bisection" and (args.left is None or args.right is None):
            parser.error("root bisection requires --left and --right")
        if args.method == "newton" and args.x0 is None:
            parser.error("root newton requires --x0")
    try:
        args.func(args)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

# Examples:
# python math_toolbox.py eval "sin(pi/6) + cos(0)"
# python math_toolbox.py linear 2 -4
# python math_toolbox.py quadratic 1 -5 6
# python math_toolbox.py system --A "2,1; 5,3" --b "1,0"
# python math_toolbox.py derivative --func "sin(x)*exp(x)" --x 1
# python math_toolbox.py integral --func "x^2 * cos(x)" --a 0 --b 1 --n 2000
# python math_toolbox.py root --func "x^3 - x - 2" --method bisection --left 1 --right 2
# python math_toolbox.py root --func "cos(x) - x" --method newton --x0 1
# python math_toolbox.py mdet --A "1,2; 3,4"
# python math_toolbox.py minv --A "4,7; 2,6"
# python math_toolbox.py mmul --A "1,2;3,4" --B "5,6;7,8"
