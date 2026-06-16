# EquiAgua v2.0 — Simulador de Distribución Equitativa de Agua

Simulador web interactivo que reparte una oferta limitada de agua entre los nodos de demanda de una ciudad **maximizando la equidad** y garantizando un **mínimo vital** por persona. Está pensado como herramienta de *tecnología cívica*: traduce un modelo de Programación Lineal en visualizaciones y en un calendario operativo de tandeo que cualquier tomador de decisiones puede leer.


## Características

- **Selector de ciudades**: carga datasets demográficos locales (Bogotá, Los Cabos, SLP, Ciudad Gótica).
- **Escenarios de estrés (What-If)**: Normal, Lluvias (+30 %), Sequía (−40 %) y Tubería Rota (−70 %).
- **Parámetros dinámicos** (sliders): consumo estándar, mínimo vital, tolerancia de equidad y oferta base.
- **Mapa de cobertura** (treemap): área = población, color = % de cobertura.
- **Equidad demográfica**: litros por persona por nodo, con líneas guía de mínimo vital, consumo ideal y banda de equidad.
- **Calendario operativo**: traducción del volumen a *días de tandeo* enteros en un ciclo de 30 días, anclados en el tiempo.
- **Manejo de infactibilidad**: cuando no hay reparto posible, una alerta de crisis explica la causa y el umbral para recuperarlo.

---

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3 |
| Optimización | [PuLP](https://coin-or.github.io/pulp/) (solver CBC) |
| Datos | pandas |
| Interfaz web | Streamlit |
| Visualización | Plotly |

---

## Instalación y ejecución

```bash
# 1. Crear y activar el entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

Para validar solo el motor matemático en consola:

```bash
python motor.py
```

---

## Estructura del proyecto

```
equiaguaV2/
├── .streamlit/
│   └── config.toml      # Tema institucional (colores y fuentes)
├── data/
│   ├── dataset_bogota.csv
│   ├── dataset_loscabos.csv
│   ├── dataset_slp.csv
│   └── dataset_gotica.csv
├── motor.py             # Motor PuLP (versión consola, optimizada)
├── app.py               # Aplicación Streamlit completa
├── requirements.txt
└── README.md
```

Cada CSV contiene dos columnas: `ID_NODO` (identificador) y `POBTOT` (población).

---

## Modelo matemático

El problema es un **programa lineal** que reparte el agua disponible minimizando el
déficit total, sujeto a equidad y a un mínimo vital.

### Conjuntos

- $I$ : conjunto de nodos de demanda, indexados por $i$.

### Parámetros

| Símbolo | Significado |
|---|---|
| $\text{Pob}_i$ | Población del nodo $i$ |
| $P = \sum_{i} \text{Pob}_i$ | Población total de la ciudad |
| $\text{Consumo}_{std}$ | Demanda ideal por persona en el ciclo (L) |
| $V_{min}$ | Mínimo vital por persona en el ciclo (L) |
| $E_{desv}$ | Tolerancia de desviación de equidad (p. ej. $0.10$) |
| $C_{Total}$ | Oferta total de agua en el ciclo de 30 días (L) |

### Variables de decisión

$$
X_i \ge 0 \quad\text{(volumen entregado al nodo } i\text{)}, \qquad
U_i \ge 0 \quad\text{(déficit del nodo } i\text{)}.
$$

### Función objetivo

Minimizar el déficit total del sistema:

$$
\min \; \sum_{i \in I} U_i
$$

### Restricciones

**1. Balance de cada nodo** — lo entregado más lo que falta iguala su demanda ideal:

$$
X_i + U_i = \text{Pob}_i \cdot \text{Consumo}_{std} \qquad \forall i \in I
$$

**2. Capacidad total** — no se puede repartir más agua que la disponible:

$$
\sum_{i \in I} X_i \le C_{Total}
$$

**3. Mínimo vital** — todo nodo recibe al menos su piso humanitario:

$$
X_i \ge \text{Pob}_i \cdot V_{min} \qquad \forall i \in I
$$

**4. y 5. Equidad (forma linealizada cruzada)** — el agua por persona de cada nodo
no se aleja más de $E_{desv}$ del promedio del sistema:

$$
X_i \cdot P \le \text{Pob}_i \cdot \Big(\sum_{j \in I} X_j\Big) \cdot (1 + E_{desv}) \qquad \forall i \in I
$$

$$
X_i \cdot P \ge \text{Pob}_i \cdot \Big(\sum_{j \in I} X_j\Big) \cdot (1 - E_{desv}) \qquad \forall i \in I
$$

### ¿Por qué la equidad está "linealizada"?

La idea natural de equidad es que el **agua por persona** de cada nodo sea parecida
al promedio de la ciudad:

$$
\frac{X_i}{\text{Pob}_i} \approx \frac{\sum_j X_j}{P}
$$

Escrita así, con $X$ en el denominador, la restricción es **no lineal** y rompe el
solver. Multiplicando ambos lados por $\text{Pob}_i \cdot P$ se obtiene la forma
**cruzada** de las restricciones 4 y 5, donde ambos lados son lineales en las
variables $X$. Es matemáticamente equivalente, pero PuLP/CBC ya la pueden resolver.

### Interpretación: la banda de equidad

$E_{desv}$ define una **banda de tolerancia** alrededor del reparto perfectamente
proporcional. Con $E_{desv}=0$ se exige igualdad estricta per cápita; con
$E_{desv}=0.10$ se admite un margen de ±10 %. Ese margen tiene sentido operativo
(la entrega real es "a bloques", los datos son imperfectos, y da holgura logística
y numérica al modelo). Es, en esencia, una **perilla de política pública**.

### Condición de factibilidad

Como cada nodo exige su mínimo vital, la suma de mínimos no puede superar la oferta.
Una condición necesaria de factibilidad es:

$$
V_{min} \le \frac{C_{Total}}{P}
$$

Si se viola (típico en el escenario **Tubería Rota**), el modelo es *Infeasible*:
no hay forma física de garantizar el mínimo vital a toda la población. La app lo
detecta y muestra una alerta de crisis con el umbral exacto para recuperar la
factibilidad.

---

## Optimización para ciudades grandes (formulación dispersa)

En la forma anterior, cada una de las $2|I|$ restricciones de equidad contiene el
término $\sum_j X_j$, es decir, referencia a **todos** los nodos. Para Ciudad Gótica
(~8000 nodos) eso genera una matriz casi densa (~$|I|^2 \approx 10^8$ términos) y el
solver se vuelve inviable.

La reformulación **equivalente** que usa el simulador:

1. **Variable auxiliar** $S = \sum_{j} X_j$, definida con una sola restricción. Así
   la equidad se reescribe con solo **dos términos** por fila:

$$
X_i \le \frac{\text{Pob}_i}{P}\,(1 + E_{desv})\,S, \qquad
X_i \ge \frac{\text{Pob}_i}{P}\,(1 - E_{desv})\,S
$$

2. **Mínimo vital y tope de demanda como cotas** de la variable (no como filas):

$$
\text{Pob}_i \cdot V_{min} \;\le\; X_i \;\le\; \text{Pob}_i \cdot \text{Consumo}_{std}
$$

3. **Capacidad como cota** de $S$: $\; 0 \le S \le C_{Total}$.

4. **Escalado numérico**: al dividir la equidad entre $P$, los coeficientes quedan
   de magnitud razonable (mejor estabilidad y velocidad en CBC).

Resultado: la matriz pasa de ~$10^8$ a ~$6\times10^4$ términos no nulos, y Gótica se
resuelve en segundos. El óptimo es idéntico al del modelo original.

---

## Del volumen al calendario: días de tandeo

El modelo entrega litros (continuos). El **Calendario Operativo** los traduce a
**días de tandeo enteros** (un día = 24 h con agua) en un ciclo de 30 días.

Los días equivalentes de cada nodo son:

$$
d_i = 30 \cdot \frac{X_i}{\text{Pob}_i \cdot \text{Consumo}_{std}} = 30 \cdot \text{cobertura}_i
$$

Redondear $d_i$ de forma ingenua puede **superar la capacidad** (la suma de redondeos
hacia arriba pide más agua de la que existe). Por eso se usa el **método del resto
mayor** (prorrateo), 100 % como post-proceso, sin tocar el modelo:

1. Asignar a cada nodo $\lfloor d_i \rfloor$ días (siempre cabe).
2. Calcular el presupuesto de agua sobrante respecto a $C_{Total}$.
3. Repartir los días extra a los nodos con mayor parte fraccionaria, mientras alcance
   el agua.

Esto garantiza que el agua del calendario nunca exceda $C_{Total}$. Los días con agua
se distribuyen además lo más uniformemente posible dentro del ciclo, para evitar
huecos largos sin servicio.

---

## Escenarios de estrés

Cada escenario multiplica la oferta base ($C_{Total}$):

| Escenario | Multiplicador |
|---|---|
| Normal | × 1.00 |
| Lluvias | × 1.30 |
| Sequía | × 0.60 |
| Tubería Rota | × 0.30 |

---

## Fuentes de los datos demográficos:

- **San Luis Potosí y Los Cabos:** Instituto Nacional de Estadística y Geografía. (2020). *Sistema para la consulta de información censal (SCITEL)* [Conjunto de datos]. Recuperado el 15 de junio de 2026, de https://www.inegi.org.mx/app/scitel/Default?ev=9

- **Bogotá:** Departamento Administrativo Nacional de Estadística. (s. f.). *Proyecciones de población de Bogotá* [Conjunto de datos]. Recuperado el 15 de junio de 2026, de https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion/proyecciones-de-poblacion-bogota

- **Ciudad Gótica:** Datos sintéticos generados por inteligencia artificial. (2026). *Ciudad Gótica: proyección de casos extremos de estrés hídrico* [Conjunto de datos sintético generado por IA]. Producido mediante un modelo de lenguaje de gran escala a partir de parámetros extremos; datos ficticios, sin correspondencia con una población real.
