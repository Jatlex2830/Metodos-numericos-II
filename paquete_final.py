'''
Desarrolladores:
Flores Villaseñor Sander Daniel
Tenorio Castelán Jatziri Berenice
Espinosa Nunez Jorge Luis Centli
'''

import numpy as np
import math
from sympy import symbols, Matrix, pprint, simplify
from scipy.optimize import broyden1
import matplotlib.pyplot as plt

x, y = symbols('x y')

def creaTabla():
    n_puntos = int(input("Escriba el número de puntos que tiene su tabla: "))
    datos = []
    print("\nIntroduzca los valores de la tabla:\n")
    for i in range(n_puntos):
        while True:
            try:
                aux1 = float(input(f"xi_{i} = "))
                aux2 = float(input(f"yi_{i} = "))
                datos.append([aux1, aux2])
                break
            except ValueError:
                print("Debe introducir solo valores numéricos.")
    datos = sorted(datos, key=lambda fila: fila[0])
    print("\nLos datos x y f(x) que introdujo son:\n")
    pprint(Matrix(datos))
    return datos

def spline():
    datos = creaTabla()
    h = [datos[i+1][0] - datos[i][0] for i in range(len(datos)-1)]
    f1 = [(datos[i+1][1] - datos[i][1]) / h[i] for i in range(len(datos)-1)]
    A = [[2 * (h[i] + h[i+1]) if i == j else h[i] if abs(i - j) == 1 else 0 for i in range(len(h)-1)] for j in range(len(h)-1)]
    f2 = [6 * (f1[i+1] - f1[i]) for i in range(len(f1)-1)]
    S_i = Matrix(A).inv() * Matrix(f2)
    S_i = [0] + list(S_i) + [0]
    polinomios = []
    for i in range(len(S_i)-1):
        a = (S_i[i+1] - S_i[i]) / (6*h[i])
        b = S_i[i] / 2
        c = f1[i] - ((S_i[i+1] + 2 * S_i[i]) * h[i] / 6)
        d = datos[i][1]
        polinomios.append((a, b, c, d, datos[i][0]))
    print("\nLos polinomios resultantes son:\n")
    for i, (a, b, c, d, xi) in enumerate(polinomios):
        print(f"g_{i}(x) = {a}(x - {xi})³ + {b}(x - {xi})² + {c}(x - {xi}) + {d}")
        print(f"\tIntervalo en ({xi}, {datos[i+1][0]})")

def productosL(t, k, longitud):
    numerador = 1.0
    denominador = 1.0
    for i in range(longitud):
        if i != k:
            numerador *= (x - t[i][0])
            denominador *= (t[k][0] - t[i][0])
    return simplify(numerador/denominador).expand()

def polinomio_lagrange(t, longitud):
    polinomio = sum(productosL(t, i, longitud) * t[i][1] for i in range(longitud))
    return polinomio

def coeficientes(tabla, grado):
    for j in range(grado):
        for i in range(grado-j):
            tabla[i].append((tabla[i+1][j+1] - tabla[i][j+1]) / (tabla[j+i+1][0] - tabla[i][0]))
    return tabla[0][1:]

def polinomio_diferencias_divididas(tabla, grado):
    a = coeficientes(tabla, grado)
    polinomio = sum(a[j] * np.prod([(x - tabla[i][0]) for i in range(j)], axis=0) for j in range(grado+1))
    return polinomio.expand()

def creaTablaInterpolacion(grado):
    datos = []
    print("\nIntroduzca los valores de la tabla:\n")
    for i in range(grado+1):
        while True:
            try:
                aux1 = float(input(f"X_{i} = "))
                aux2 = float(input(f"f(x)_{i} = "))
                datos.append([aux1, aux2])
                break
            except ValueError:
                print("Debe introducir solo valores numéricos.")
    datos = sorted(datos, key=lambda fila: fila[0])
    print("\nLos datos x y f(x) que introdujo son:\n")
    pprint(Matrix(datos))
    return datos

def interpolacion():
    print("Bienvenido. ¿Qué método de interpolación quiere usar?")
    while True:
        try:
            opcion = int(input("\n1. Lagrange \n2. Diferencias divididas\n"))
            if opcion not in [1, 2]:
                raise ValueError()
            break
        except ValueError:
            print("Debe introducir solo valores válidos.")
    while True:
        try:
            grado = int(input("¿De qué grado busca el polinomio? "))
            if grado < 1:
                raise ValueError()
            break
        except ValueError:
            print("Debe introducir solo valores enteros.")
    while True:
        try:
            opcion_evaluar = int(input("\n¿Quiere evaluarlos en un punto? 1.Sí  2.No\t"))
            if opcion_evaluar not in [1, 2]:
                raise ValueError()
            break
        except ValueError:
            print("Debe introducir solo valores válidos.")
    datos = creaTablaInterpolacion(grado)
    if opcion == 1:
        polinomio = polinomio_lagrange(datos, len(datos))
        print(f"\nEl polinomio de grado {grado} con método de Lagrange es:\n\n{polinomio}")
        if opcion_evaluar == 1:
            while True:
                try:
                    valorX = float(input("Introduzca el valor de la X buscada: "))
                    if valorX < datos[0][0] or valorX > datos[-1][0]:
                        raise ValueError()
                    print(f"La función evaluada en el valor {valorX} es: {polinomio.subs({x:valorX})}")
                    break
                except ValueError:
                    print("Por favor introduce un valor que esté dentro del rango de los datos a interpolar")
    elif opcion == 2:
        polinomio = polinomio_diferencias_divididas(datos, len(datos)-1)
        print(f"\nEl polinomio de grado {grado} con método de diferencias divididas es:\n\n{polinomio}")
        if opcion_evaluar == 1:
            while True:
                try:
                    valorX = float(input("Introduzca el valor de la X buscada: "))
                    if valorX < datos[0][0] or valorX > datos[-1][0]:
                        raise ValueError()
                    print(f"La función evaluada en el valor {valorX} es: {polinomio.subs({x:valorX})}")
                    break
                except ValueError:
                    print("Por favor introduce un valor que esté dentro del rango de los datos a interpolar")

def f1(x):
    return x**4 * (math.sqrt(3 + 2 * x**2) / 3)

def f2(x):
    return x**5 / ((x**2 + 4)**(1/5))

def simpson_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    sum_odd = sum(f(a + h * i) for i in range(1, n, 2))
    sum_even = sum(f(a + h * i) for i in range(2, n, 2))
    return (h / 3) * (f(a) + f(b) + 4 * sum_odd + 2 * sum_even)

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    sum_mid = sum(f(a + h * i) for i in range(1, n))
    return (h / 2) * (f(a) + f(b) + 2 * sum_mid)

def numerical_derivative(f, a, b, n):
    h = (b - a) / n
    xi = [a + i * h for i in range(n+1)]
    fi = [f(xi[i]) for i in range(n+1)]
    f_prime = [(fi[i+1] - fi[i-1]) / (2 * h) if 0 < i < n else (fi[i+1] - fi[i]) / h if i == 0 else (fi[i] - fi[i-1]) / h for i in range(n+1)]
    return xi, fi, f_prime

def integracion_numerica():
    while True:
        print("Seleccione la función a integrar:")
        print("1. f(x) = x^4 * ((sqrt(3 + 2x^2) / 3))")
        print("2. f(x) = x^5 / ((x^2 + 4)^(1/5))")
        print("3. Salir")
        opcion = input("Ingrese el número de su elección: ")
        if opcion == "3":
            break
        if opcion not in ["1", "2"]:
            print("Opción no válida. Intente de nuevo.")
            continue
        if opcion == "1":
            f = f1
        else:
            f = f2

        a = float(input("Ingrese el límite inferior de integración (a): "))
        b = float(input("Ingrese el límite superior de integración (b): "))
        n = int(input("Ingrese el número de intervalos (n): "))
        while n <= 0:
            print("El número de intervalos debe ser mayor que 0. Inténtelo de nuevo.")
            n = int(input("Ingrese el número de intervalos (n): "))

        print("Seleccione el método de integración:")
        print("1. Regla de Simpson")
        print("2. Regla del Trapecio")
        metodo = input("Ingrese el número de su elección: ")

        if metodo == "1":
            resultado = simpson_rule(f, a, b, n)
        elif metodo == "2":
            resultado = trapezoidal_rule(f, a, b, n)
        else:
            print("Opción no válida. Intente de nuevo.")
            continue

        print(f"El resultado de la integración es: {resultado}")

def derivada_numerica():
    print("Bienvenido a la derivada numérica.")
    while True:
        try:
            print("Seleccione una función para derivar:")
            print("1. f(x) = x^4 * ((sqrt(3 + 2x^2) / 3))")
            print("2. f(x) = x^5 / ((x^2 + 4)^(1/5))")
            opcion = int(input("Ingrese el número de su elección: "))
            if opcion not in [1, 2]:
                raise ValueError("Opción no válida.")
            break
        except ValueError as e:
            print(e)

    if opcion == 1:
        f = f1
    elif opcion == 2:
        f = f2

    while True:
        try:
            a = float(input("Ingrese el límite inferior (a): "))
            b = float(input("Ingrese el límite superior (b): "))
            if a >= b:
                raise ValueError("El límite inferior debe ser menor que el límite superior.")
            break
        except ValueError as e:
            print(e)

    while True:
        try:
            n = int(input("Ingrese el número de intervalos (n): "))
            if n <= 0:
                raise ValueError("El número de intervalos debe ser mayor que 0.")
            break
        except ValueError as e:
            print(e)

    xi, fi, f_prime = numerical_derivative(f, a, b, n)

    print("Resultados de la derivada numérica:")
    print("xi\tf(xi)\tf'(xi)")
    for i in range(len(xi)):
        print(f"{xi[i]:.6f}\t{fi[i]:.6f}\t{f_prime[i]:.6f}")

# Funciones para el método de mínimos cuadrados

def crear_tabla():
    n_puntos = int(input("Escriba el número de puntos que tiene su tabla: "))
    datos = []
    print("\nIntroduzca los valores de la tabla:\n")
    for i in range(n_puntos):
        while True:
            try:
                xi = float(input(f"xi_{i} = "))
                yi = float(input(f"yi_{i} = "))
                datos.append([xi, yi])
                break
            except ValueError:
                print("Debe introducir solo valores numéricos.")
    datos = np.array(sorted(datos, key=lambda fila: fila[0]))
    return datos

def minimos_cuadrados(datos, grado):
    X = np.vander(datos[:, 0], N=grado+1, increasing=True)
    y = datos[:, 1]
    coeficientes = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeficientes

def mostrar_resultados(datos, coeficientes, grado):
    x_fit = np.linspace(min(datos[:, 0]), max(datos[:, 0]), 1000)
    y_fit = sum(coeficientes[i] * x_fit**i for i in range(grado+1))

    print("\nCoeficientes del polinomio ajustado:")
    for i, coef in enumerate(coeficientes):
        print(f"a_{i} = {coef}")

    plt.scatter(datos[:, 0], datos[:, 1], color='red', label='Datos')
    plt.plot(x_fit, y_fit, label=f'Polinomio de grado {grado}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajuste por Mínimos Cuadrados')
    plt.legend()
    plt.show()

def ajuste_minimos_cuadrados():
    print("Método de Mínimos Cuadrados")
    datos = crear_tabla()
    while True:
        try:
            grado = int(input("Ingrese el grado del polinomio de ajuste: "))
            if grado < 0:
                raise ValueError()
            break
        except ValueError:
            print("Debe introducir un número entero no negativo.")

    coeficientes = minimos_cuadrados(datos, grado)
    mostrar_resultados(datos, coeficientes, grado)

def menu():
    while True:
        print("\nSeleccione una opción:")
        print("1. Interpolación")
        print("2. Spline cúbico")
        print("3. Integración numérica")
        print("4. Derivada numérica")
        print("5. Mínimos cuadrados")
        print("6. Salir")
        opcion = input("Ingrese el número de su elección: ")
        if opcion == "1":
            interpolacion()
        elif opcion == "2":
            spline()
        elif opcion == "3":
            integracion_numerica()
        elif opcion == "4":
            derivada_numerica()
        elif opcion == "5":
            ajuste_minimos_cuadrados()
        elif opcion == "6":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    menu()