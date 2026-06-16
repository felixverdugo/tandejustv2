"""
EquiAgua v2.0 - Motor matematico OPTIMIZADO (PuLP)
--------------------------------------------------
Misma formulacion del modelo, pero reescrita para escalar a ciudades
grandes (p. ej. Ciudad Gotica, ~8000 nodos).

¿Que se optimizo y porque?
  1) Variable auxiliar S = sum(X_i), definida UNA sola vez.
     -> Las 2N restricciones de equidad pasan de N+1 terminos a solo 2.
        (Antes la matriz era densa: ~N^2 terminos. Ahora es dispersa: ~N.)
  2) Minimo vital y tope de demanda como COTAS de variable (bounds),
     no como filas de restriccion. (Ahorra 2N filas.)
  3) Escalado numerico: la equidad se divide entre Poblacion_Total para
     que los coeficientes sean de magnitud razonable (mejor para CBC).

El resultado matematico es identico al modelo original.

Ejecutar:   python motor.py
"""

import time
import pandas as pd
import pulp

# =====================================================================
# 1) PARAMETROS
# =====================================================================
RUTA_CSV    = "data/dataset_gotica.csv"  # <-- la ciudad grande (~8000 nodos)
CONSUMO_STD = 3000.0   # Demanda ideal por persona en el ciclo (L / 30 dias)
V_MIN       = 1000.0   # Minimo vital por persona en el ciclo (L)
E_DESV      = 0.10     # Tolerancia de desviacion de equidad (10%)
FRACCION_C  = 0.80     # Oferta = 80% de la demanda total

# =====================================================================
# 2) CARGA DE DATOS
# =====================================================================
df = pd.read_csv(RUTA_CSV, dtype={"ID_NODO": str})
df = df.groupby("ID_NODO", as_index=False)["POBTOT"].sum()
df = df[df["POBTOT"] > 0].reset_index(drop=True)

nodos     = df["ID_NODO"].tolist()
poblacion = dict(zip(df["ID_NODO"], df["POBTOT"].astype(float)))
P         = sum(poblacion.values())
DEMANDA_TOT = sum(p * CONSUMO_STD for p in poblacion.values())
C_TOTAL     = FRACCION_C * DEMANDA_TOT

print("=" * 60)
print(f"Ciudad: {RUTA_CSV}")
print(f"Nodos de demanda : {len(nodos):,}")
print(f"Poblacion total  : {P:,.0f} hab")
print(f"Oferta C_Total   : {C_TOTAL:,.0f} L  ({FRACCION_C*100:.0f}% de la demanda)")
print("=" * 60)

# =====================================================================
# 3) MODELO PuLP (formulacion dispersa)
# =====================================================================
t0 = time.perf_counter()
modelo = pulp.LpProblem("EquiAgua", pulp.LpMinimize)

# --- Variables ---
# X_i con cotas: minimo vital <= X_i <= demanda del nodo
#   (asi el "minimo vital" y "U_i >= 0" no necesitan filas de restriccion)
X = {i: pulp.LpVariable(f"X_{i}",
                        lowBound=poblacion[i] * V_MIN,
                        upBound=poblacion[i] * CONSUMO_STD)
     for i in nodos}
# U_i = deficit (para la funcion objetivo y el reporte)
U = {i: pulp.LpVariable(f"U_{i}", lowBound=0) for i in nodos}
# S = agua total entregada, acotada por la capacidad
S = pulp.LpVariable("S", lowBound=0, upBound=C_TOTAL)

# --- Funcion objetivo: minimizar el deficit total ---
modelo += pulp.lpSum(U.values()), "Minimizar_Deficit_Total"

# --- Definicion de S (UNICA fila densa del modelo) ---
modelo += S == pulp.lpSum(X.values()), "Suma_Total"

# --- Restricciones por nodo (todas dispersas) ---
for i in nodos:
    pi = poblacion[i]
    # Balance:  X_i + U_i = demanda_i
    modelo += X[i] + U[i] == pi * CONSUMO_STD, f"Balance_{i}"
    # Equidad (escalada entre P): coeficientes pequenos, solo 2 terminos
    #   X_i <= (Pob_i/P) * S * (1+E)   y   X_i >= (Pob_i/P) * S * (1-E)
    modelo += X[i] <= (pi / P) * (1 + E_DESV) * S, f"EquidadSup_{i}"
    modelo += X[i] >= (pi / P) * (1 - E_DESV) * S, f"EquidadInf_{i}"

t_build = time.perf_counter() - t0
print(f"Modelo construido en {t_build:.1f} s "
      f"({len(modelo.variables()):,} variables, "
      f"{len(modelo.constraints):,} restricciones)")

# =====================================================================
# 4) RESOLVER
# =====================================================================
t0 = time.perf_counter()
modelo.solve(pulp.PULP_CBC_CMD(msg=False, threads=4, presolve=True))
t_solve = time.perf_counter() - t0
estado = pulp.LpStatus[modelo.status]
print(f"Resuelto en {t_solve:.1f} s  ->  estado: {estado}\n")

if estado == "Optimal":
    df["X_entregado"] = df["ID_NODO"].map(lambda i: X[i].value())
    df["U_deficit"]   = df["ID_NODO"].map(lambda i: U[i].value())
    df["demanda"]     = df["POBTOT"] * CONSUMO_STD
    df["cobertura_%"] = 100 * df["X_entregado"] / df["demanda"]

    total_X = df["X_entregado"].sum()
    total_U = df["U_deficit"].sum()
    print(f"Agua total asignada : {total_X:,.0f} L")
    print(f"Deficit total (obj) : {total_U:,.0f} L")
    print(f"Cobertura global    : {100*total_X/DEMANDA_TOT:.1f}%")
    print("\n--- Muestra de 8 nodos ---")
    print(df.head(8).to_string(
        index=False,
        formatters={
            "POBTOT":      "{:,.0f}".format,
            "X_entregado": "{:,.0f}".format,
            "U_deficit":   "{:,.0f}".format,
            "demanda":     "{:,.0f}".format,
            "cobertura_%": "{:.1f}".format,
        },
    ))
else:
    print(">> Sin solucion optima (revisar parametros o factibilidad).")