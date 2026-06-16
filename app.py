"""
EquiAgua v2.0 - FASE 6: Manejo de infactibilidad
------------------------------------------------
Cierra el simulador. Cuando el modelo resulta Infeasible (p. ej.
"Tuberia Rota"), en vez de un mensaje tecnico se muestra una ALERTA DE
CRISIS con el diagnostico de la causa y como recuperar la factibilidad.
  Tab 1 - Treemap de equidad (area = poblacion, color = % cobertura)
  Tab 2 - Agua por persona por nodo con lineas guia (plotly)
  Tab 3 - Calendario operativo: dias de tandeo + calendario en el tiempo

Ejecutar:   streamlit run app.py
"""

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st

# =====================================================================
# CONFIGURACION GENERAL DE LA PAGINA
# =====================================================================
st.set_page_config(
    page_title="EquiAgua v2.0",
    page_icon="💧",
    layout="wide",
)

CIUDADES = {
    "Bogotá":    "data/dataset_bogota.csv",
    "Los Cabos": "data/dataset_loscabos.csv",
    "SLP":       "data/dataset_slp.csv",
    "Gótica":    "data/dataset_gotica.csv",
}

ESCENARIOS = {
    "Normal":              1.00,
    "Lluvias (+30%)":      1.30,
    "Sequía (-40%)":       0.60,
    "Tubería Rota (-70%)": 0.30,
}

VERDE = "#2ecc71"
ROJO  = "#e74c3c"


@st.cache_data
def cargar_ciudad(ruta: str) -> pd.DataFrame:
    """Carga y limpia el CSV de una ciudad (agrupa IDs repetidos)."""
    df = pd.read_csv(ruta, dtype={"ID_NODO": str})
    df = df.groupby("ID_NODO", as_index=False)["POBTOT"].sum()
    df = df[df["POBTOT"] > 0].reset_index(drop=True)
    return df


# =====================================================================
# MOTOR MATEMATICO (PuLP)
# =====================================================================
def resolver_equiagua(df, consumo_std, v_min, e_desv, c_total):
    """Construye y resuelve el modelo (formulacion DISPERSA, escala a miles
    de nodos). Devuelve (estado, df_resultado).

    Optimizaciones vs. la version densa:
      - S = sum(X) como variable: la equidad usa 2 terminos por fila (no N).
      - minimo vital y tope de demanda como cotas de X (no como filas).
      - equidad escalada entre P (mejor comportamiento numerico).
    El optimo es identico al modelo original."""
    nodos     = df["ID_NODO"].tolist()
    poblacion = dict(zip(df["ID_NODO"], df["POBTOT"].astype(float)))
    P         = sum(poblacion.values())

    modelo = pulp.LpProblem("EquiAgua", pulp.LpMinimize)

    # X_i acotada: minimo vital <= X_i <= demanda del nodo
    X = {i: pulp.LpVariable(f"X_{i}",
                            lowBound=poblacion[i] * v_min,
                            upBound=poblacion[i] * consumo_std)
         for i in nodos}
    U = {i: pulp.LpVariable(f"U_{i}", lowBound=0) for i in nodos}
    S = pulp.LpVariable("S", lowBound=0, upBound=c_total)   # agua total entregada

    modelo += pulp.lpSum(U.values()), "Minimizar_Deficit_Total"
    modelo += S == pulp.lpSum(X.values()), "Suma_Total"     # unica fila densa

    for i in nodos:
        pi = poblacion[i]
        modelo += X[i] + U[i] == pi * consumo_std, f"Balance_{i}"
        modelo += X[i] <= (pi / P) * (1 + e_desv) * S, f"EquidadSup_{i}"
        modelo += X[i] >= (pi / P) * (1 - e_desv) * S, f"EquidadInf_{i}"

    try:
        modelo.solve(pulp.PULP_CBC_CMD(msg=False, threads=4, presolve=True))
        estado = pulp.LpStatus[modelo.status]
    except Exception as exc:                       # fallo del solver
        return f"Error: {exc}", df.copy()

    res = df.copy()
    if estado == "Optimal":
        res["X_entregado"] = res["ID_NODO"].map(lambda i: X[i].value())
        res["U_deficit"]   = res["ID_NODO"].map(lambda i: U[i].value())
        res["demanda"]     = res["POBTOT"] * consumo_std
        res["cobertura_%"] = 100 * res["X_entregado"] / res["demanda"]
        res["estatus"]     = np.where(res["U_deficit"] > 1e-6,
                                      "Con déficit", "Satisfecho")
    return estado, res


def diagnosticar_infactibilidad(pob_total, consumo_std, v_min, c_total):
    """Devuelve una lista de causas probables (en lenguaje simple) de que
    el modelo sea infactible, segun las restricciones del problema."""
    causas = []
    min_vital_total = pob_total * v_min

    if c_total <= 0:
        causas.append("La oferta C_Total es cero o negativa: no hay agua que repartir.")
    if v_min > consumo_std:
        causas.append(
            f"El mínimo vital ({v_min:,.0f} L/persona) es mayor que el consumo ideal "
            f"({consumo_std:,.0f} L). El piso no puede superar a la demanda total.")
    if c_total > 0 and min_vital_total > c_total:
        causas.append(
            f"Garantizar el mínimo vital a toda la población exige "
            f"{min_vital_total:,.0f} L, pero la oferta disponible es solo "
            f"{c_total:,.0f} L (faltan {min_vital_total - c_total:,.0f} L).")
    if not causas:
        causas.append(
            "La combinación de mínimo vital, equidad y capacidad no deja una región "
            "factible. Suele resolverse aflojando la tolerancia de equidad o el mínimo "
            "vital.")
    return causas


# =====================================================================
# VISUALIZACIONES (FASE 4)
# =====================================================================
def fig_treemap(res: pd.DataFrame) -> go.Figure:
    """Treemap de equidad: cada rectangulo es un nodo.
    Area = poblacion | color = % de cobertura (rojo->amarillo->verde).
    Muestra TODOS los nodos a la vez (escala a miles)."""
    d = res.copy()
    fig = px.treemap(
        d,
        path=[px.Constant("Macrotanque"), "ID_NODO"],
        values="POBTOT",
        color="cobertura_%",
        color_continuous_scale="RdYlGn",
        range_color=(0, 100),
        custom_data=["POBTOT", "X_entregado", "U_deficit", "cobertura_%"],
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="white")),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Población: %{customdata[0]:,.0f}<br>"
            "Entregado: %{customdata[1]:,.0f} L<br>"
            "Déficit: %{customdata[2]:,.0f} L<br>"
            "Cobertura: %{customdata[3]:.1f}%<extra></extra>"
        ),
    )
    fig.update_layout(
        title="Mapa de equidad — área = población · color = % de cobertura",
        height=620, margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title="Cobertura %"),
    )
    return fig


def fig_agua_por_persona(res: pd.DataFrame, v_min: float,
                         consumo_std: float, e_desv: float) -> go.Figure:
    """Agua por persona en cada nodo (ordenado de menor a mayor), con
    lineas guia: minimo vital (piso), consumo ideal (meta), promedio y
    banda de equidad sombreada. El color refleja la cobertura."""
    d = res.copy()
    d["lts_persona"] = d["X_entregado"] / d["POBTOT"]
    d = d.sort_values("lts_persona").reset_index(drop=True)
    d["rank"] = range(1, len(d) + 1)

    promedio = d["X_entregado"].sum() / d["POBTOT"].sum()   # per capita realizado

    fig = go.Figure()

    # --- Banda de equidad alrededor del promedio (+/- E_desv) ---
    fig.add_hrect(
        y0=promedio * (1 - e_desv), y1=promedio * (1 + e_desv),
        fillcolor="rgba(52,152,219,0.12)", line_width=0,
        annotation_text="Banda de equidad (±E_desv)",
        annotation_position="top left",
    )

    # --- Puntos: litros por persona, color = cobertura ---
    fig.add_trace(go.Scattergl(
        x=d["rank"], y=d["lts_persona"], mode="markers",
        marker=dict(size=6, color=d["cobertura_%"], colorscale="RdYlGn",
                    cmin=0, cmax=100, colorbar=dict(title="Cobertura %"),
                    line=dict(width=0)),
        text=d["ID_NODO"], customdata=d["cobertura_%"],
        hovertemplate=("Nodo %{text}<br>%{y:,.0f} L/persona<br>"
                       "Cobertura: %{customdata:.1f}%<extra></extra>"),
        name="Litros por persona",
    ))

    # --- Lineas guia ---
    fig.add_hline(y=consumo_std, line=dict(color=VERDE, dash="dash"),
                  annotation_text=f"Consumo ideal ({consumo_std:,.0f} L)",
                  annotation_position="top right")
    fig.add_hline(y=v_min, line=dict(color=ROJO, dash="dash"),
                  annotation_text=f"Mínimo vital ({v_min:,.0f} L)",
                  annotation_position="bottom right")
    fig.add_hline(y=promedio, line=dict(color="gray", dash="dot"),
                  annotation_text=f"Promedio ({promedio:,.0f} L)",
                  annotation_position="bottom left")

    fig.update_layout(
        title="Agua por persona en cada nodo (ordenado)",
        xaxis_title="Nodos (ordenados de menor a mayor L/persona)",
        yaxis_title="Litros por persona — ciclo 30 días",
        yaxis_range=[0, consumo_std * 1.08],
        height=620, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


# =====================================================================
# CALENDARIO OPERATIVO (FASE 5)  -- post-proceso, el modelo no cambia
# =====================================================================
CICLO = 30  # dias del ciclo


def calcular_calendario(res: pd.DataFrame, c_total: float) -> pd.DataFrame:
    """Traduce X_entregado a DIAS DE TANDEO enteros (0..30) por el metodo
    del resto mayor, sin superar c_total. Cada 'dia' de un nodo cuesta
    demanda_i / 30 litros, asi que se reparte el residuo en litros."""
    d = res.copy().reset_index(drop=True)

    dia_lts   = d["demanda"] / CICLO                       # litros que cuesta 1 dia/nodo
    dias_cont = CICLO * d["X_entregado"] / d["demanda"]    # = 30 * cobertura
    piso      = np.floor(dias_cont).astype(int)
    frac      = dias_cont - piso

    # Presupuesto de agua para los dias extra (los pisos siempre caben)
    budget = c_total - float((piso * dia_lts).sum())

    dias = piso.copy()
    # Repartir el residuo: fracciones mas altas primero, mientras alcance el agua
    for i in frac.sort_values(ascending=False).index:
        if frac[i] <= 0:
            break
        if dias[i] >= CICLO:
            continue
        if dia_lts[i] <= budget + 1e-6:
            dias[i] += 1
            budget  -= dia_lts[i]

    d["dias_tandeo"]  = dias.astype(int)
    d["dias_sin_agua"] = CICLO - d["dias_tandeo"]
    # Cada cuantos dias, en promedio, recibe agua (para lenguaje simple)
    d["cada_n_dias"] = np.where(d["dias_tandeo"] > 0,
                                CICLO / d["dias_tandeo"], np.inf)
    return d


DIAS_SEMANA = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def dias_con_agua_set(d: int) -> set:
    """Reparte de forma uniforme 'd' dias con agua dentro del ciclo (1..30)."""
    return {j for j in range(1, CICLO + 1)
            if (j * d) // CICLO > ((j - 1) * d) // CICLO}


def fig_calendario_nodo(d_agua: int, fecha_inicio, etiqueta: str) -> go.Figure:
    """Calendario de 30 dias (un dia = 24h) alineado a la semana, desde
    fecha_inicio. Verde = dia completo con agua | gris = dia seco."""
    aguas  = dias_con_agua_set(int(d_agua))
    fechas = [fecha_inicio + datetime.timedelta(days=k) for k in range(CICLO)]
    offset = fechas[0].weekday()                 # lunes = 0
    n_filas = -(-(offset + CICLO) // 7)          # techo de la division

    z  = [[None] * 7 for _ in range(n_filas)]
    cd = [[""]   * 7 for _ in range(n_filas)]
    anotaciones = []
    for k, f in enumerate(fechas):
        fila, col = divmod(offset + k, 7)
        con_agua  = (k + 1) in aguas
        z[fila][col]  = 1 if con_agua else 0
        cd[fila][col] = f"{f.strftime('%d/%m/%Y')} · {'Con agua' if con_agua else 'Día seco'}"
        anotaciones.append(dict(x=col, y=fila, text=str(f.day),
                                showarrow=False,
                                font=dict(size=12, color="#1f2d3d")))

    fig = go.Figure(go.Heatmap(
        z=z, customdata=cd, zmin=0, zmax=1,
        colorscale=[[0.0, "#dfe6ea"], [0.5, "#dfe6ea"],
                    [0.5, "#2ecc71"], [1.0, "#2ecc71"]],
        showscale=False, xgap=4, ygap=4, hoverongaps=False,
        hovertemplate="%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Calendario de tandeo — {etiqueta}",
        annotations=anotaciones,
        xaxis=dict(tickmode="array", tickvals=list(range(7)),
                   ticktext=DIAS_SEMANA, side="top",
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", showticklabels=False,
                   showgrid=False, zeroline=False),
        height=120 + 58 * n_filas, margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =====================================================================
# BARRA LATERAL: CONTROLES
# =====================================================================
st.sidebar.title("Panel de control")

ciudad = st.sidebar.selectbox("Selecciona la ciudad", list(CIUDADES.keys()))

st.sidebar.markdown("### Escenario de estrés")
escenario = st.sidebar.radio("Simulación de crisis", list(ESCENARIOS.keys()))

st.sidebar.markdown("### Parámetros del modelo (ciclo de 30 días)")
consumo_std = st.sidebar.slider("Consumo estándar (L/persona)", 1000, 6000, 3000, 100)
v_min       = st.sidebar.slider("Mínimo vital (L/persona)",         0, 3000, 1000, 50)
e_desv      = st.sidebar.slider("Tolerancia de equidad (E_desv)", 0.0, 0.50, 0.10, 0.01)
oferta_pct  = st.sidebar.slider("Oferta base (% de la demanda)",   30, 120, 80, 5)

# =====================================================================
# CARGA DE DATOS Y CALCULO DE LA OFERTA SEGUN ESCENARIO
# =====================================================================
df = cargar_ciudad(CIUDADES[ciudad])

pob_total   = float(df["POBTOT"].sum())
demanda_tot = pob_total * consumo_std
c_base      = (oferta_pct / 100.0) * demanda_tot
c_total     = c_base * ESCENARIOS[escenario]

# =====================================================================
# AREA PRINCIPAL
# =====================================================================
st.title("EquiAgua v2.0 — Simulador de Distribución Equitativa de Agua")
st.caption(f"Ciudad: **{ciudad}**  |  Escenario: **{escenario}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodos de demanda", f"{len(df):,}")
c2.metric("Población total",  f"{pob_total:,.0f}")
c3.metric("Demanda ideal",    f"{demanda_tot:,.0f} L")
c4.metric("Oferta C_Total",   f"{c_total:,.0f} L",
          delta=f"{(ESCENARIOS[escenario]-1)*100:+.0f}% vs base")

if st.button("Calcular distribución equitativa", type="primary"):
    with st.spinner("Resolviendo modelo de programación lineal..."):
        estado, resultado = resolver_equiagua(df, consumo_std, v_min, e_desv, c_total)
    st.session_state["estado"]    = estado
    st.session_state["resultado"] = resultado
    st.session_state["contexto"]  = f"{ciudad} · {escenario}"
    st.session_state["params"]    = {"v_min": v_min, "consumo_std": consumo_std,
                                     "e_desv": e_desv, "c_total": c_total,
                                     "pob_total": pob_total}

st.divider()

# =====================================================================
# RESULTADOS
# =====================================================================
estado    = st.session_state.get("estado")
resultado = st.session_state.get("resultado")

# Blindaje: si hay un resultado CON solucion pero sin la columna 'estatus'
# (version anterior en memoria), la reconstruimos. Si el resultado es de un
# caso infactible no tiene 'U_deficit', asi que NO se toca.
if (resultado is not None
        and "U_deficit" in resultado.columns
        and "estatus" not in resultado.columns):
    resultado = resultado.copy()
    resultado["estatus"] = np.where(resultado["U_deficit"] > 1e-6,
                                    "Con déficit", "Satisfecho")
    st.session_state["resultado"] = resultado

if estado is None:
    st.info("Configura los parámetros y pulsa **Calcular distribución equitativa**.")
elif estado != "Optimal":
    pp = st.session_state.get("params", {
        "v_min": v_min, "consumo_std": consumo_std,
        "c_total": c_total, "pob_total": pob_total})

    if str(estado).startswith("Error"):
        st.error("**Error al ejecutar el solver.**")
        st.code(estado, language="text")
        st.markdown("Revisa que PuLP/CBC estén bien instalados "
                    "(`pip install pulp`) y vuelve a intentar.")
    else:
        st.error("**Crisis crítica: no existe un reparto factible "
                 "con estos parámetros.**")
        st.markdown(
            f"El solver devolvió `{estado}`. Con la oferta y las reglas actuales es "
            "**físicamente imposible** cumplir a la vez el mínimo vital, la equidad "
            "y la capacidad. En términos de política pública: la infraestructura "
            "disponible no alcanza para garantizar el mínimo vital a toda la población.")

        causas = diagnosticar_infactibilidad(
            pp["pob_total"], pp["consumo_std"], pp["v_min"], pp["c_total"])
        st.markdown("**Causa(s) más probable(s):**")
        for c in causas:
            st.markdown(f"- {c}")

        if pp["pob_total"] > 0 and pp["c_total"] > 0:
            umbral = pp["c_total"] / pp["pob_total"]
            st.info(
                "**Cómo recuperar la factibilidad (elige una):**\n\n"
                f"1. Baja el **Mínimo vital** a **≤ {umbral:,.0f} L/persona**.\n"
                "2. Sube la **Oferta base (%)** o cambia a un escenario menos severo.\n"
                "3. Afloja la **Tolerancia de equidad (E_desv)**.")
else:
    total_X = resultado["X_entregado"].sum()
    total_U = resultado["U_deficit"].sum()
    cobertura_glob = 100 * total_X / resultado["demanda"].sum()
    nodos_deficit  = int((resultado["U_deficit"] > 1e-6).sum())

    st.success(f"Solución óptima encontrada — {st.session_state.get('contexto','')}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Agua asignada",     f"{total_X:,.0f} L")
    r2.metric("Déficit total",     f"{total_U:,.0f} L")
    r3.metric("Cobertura global",  f"{cobertura_glob:.1f}%")
    r4.metric("Nodos con déficit", f"{nodos_deficit:,} / {len(resultado):,}")

    # Parametros con los que se resolvio (fallback a los sliders actuales)
    p = st.session_state.get("params",
                             {"v_min": v_min, "consumo_std": consumo_std,
                              "e_desv": e_desv, "c_total": c_total})

    tab1, tab2, tab3 = st.tabs(
        ["Mapa de cobertura", "Equidad demográfica", "Calendario operativo"]
    )

    with tab1:
        st.caption("Cada rectángulo es un nodo. Tamaño = población · "
                   "color = % de cobertura (rojo = déficit, verde = satisfecho).")
        st.plotly_chart(fig_treemap(resultado), use_container_width=True)

    with tab2:
        st.caption("Litros por persona en cada nodo, ordenados. Sobre el **mínimo "
                   "vital** y cerca del **consumo ideal** = mejor; la banda azul es "
                   "la tolerancia de equidad.")
        st.plotly_chart(
            fig_agua_por_persona(resultado, p["v_min"], p["consumo_std"], p["e_desv"]),
            use_container_width=True,
        )

    with tab3:
        st.caption("Traducción del volumen asignado a **días de tandeo** en un ciclo "
                   "de 30 días (método del resto mayor: respeta la capacidad total).")
        cal = calcular_calendario(resultado, p["c_total"])

        prom_dias = float((cal["dias_tandeo"] * cal["POBTOT"]).sum() / cal["POBTOT"].sum())
        agua_cal  = float((cal["dias_tandeo"] * cal["demanda"] / CICLO).sum())
        pob_critica = int(cal.loc[cal["dias_tandeo"] < 15, "POBTOT"].sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Días de agua promedio", f"{prom_dias:.1f} / 30",
                  help="Promedio ponderado por población.")
        k2.metric("Agua del calendario", f"{agua_cal:,.0f} L",
                  delta=f"{agua_cal - p['c_total']:,.0f} L vs C_Total",
                  delta_color="off")
        k3.metric("Población con < 15 días", f"{pob_critica:,}")

        st.markdown("**Lectura en lenguaje simple:** un nodo con *24 días de tandeo* "
                    "recibe agua 24 de cada 30 días completos (aprox. 4 de cada 5 días); "
                    "uno con *10 días* solo tiene servicio 1 de cada 3 días. Los días sin "
                    "agua son los que el barrio debe cubrir con almacenamiento o pipas.")

        # --- Calendario anclado en el tiempo (un dia = 24h con agua) ---
        st.markdown("#### Calendario en el tiempo")
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            fecha_inicio = st.date_input("Fecha de inicio del ciclo",
                                         value=datetime.date.today())
        with cc2:
            cal_ord = cal.sort_values("POBTOT", ascending=False)
            opciones = cal_ord["ID_NODO"].tolist()
            dias_por_id = dict(zip(cal["ID_NODO"], cal["dias_tandeo"]))
            sel = st.selectbox(
                "Nodo a planear (ordenados por población)", opciones,
                format_func=lambda i: f"Nodo {i} — {int(dias_por_id[i])} días con agua",
            )
        fila = cal.loc[cal["ID_NODO"] == sel].iloc[0]
        st.plotly_chart(
            fig_calendario_nodo(fila["dias_tandeo"], fecha_inicio, f"Nodo {sel}"),
            use_container_width=True,
        )
        st.caption(
            f"Nodo {sel}: **{int(fila['dias_tandeo'])} días con agua** y "
            f"**{int(fila['dias_sin_agua'])} días secos** en el ciclo. "
            "Los días con agua se reparten lo más uniforme posible para no dejar "
            "huecos largos sin servicio.")

        st.markdown("#### Tabla por nodo")
        tabla = cal[["ID_NODO", "POBTOT", "cobertura_%",
                     "dias_tandeo", "dias_sin_agua"]].copy()
        tabla = tabla.rename(columns={
            "POBTOT": "Población", "cobertura_%": "Cobertura %",
            "dias_tandeo": "Días con agua", "dias_sin_agua": "Días sin agua"})
        st.dataframe(tabla.sort_values("Días con agua"),
                     use_container_width=True, hide_index=True)

    with st.expander("Ver resultados por nodo"):
        st.dataframe(
            resultado[["ID_NODO", "POBTOT", "demanda",
                       "X_entregado", "U_deficit", "cobertura_%", "estatus"]],
            use_container_width=True,
        )

# =====================================================================
# FUENTES DE DATOS (APA 7) -- siempre visible al pie de la app
# =====================================================================
st.divider()
st.subheader("Fuentes de los datos demográficos:")
st.markdown(
    "- **San Luis Potosí y Los Cabos:** Instituto Nacional de Estadística y "
    "Geografía. (2020). *Sistema para la consulta de información censal "
    "(SCITEL)* [Conjunto de datos]. Recuperado el 15 de junio de 2026, de "
    "https://www.inegi.org.mx/app/scitel/Default?ev=9"
)
st.markdown(
    "- **Bogotá:** Departamento Administrativo Nacional de Estadística. (s. f.). "
    "*Proyecciones de población de Bogotá* [Conjunto de datos]. Recuperado el 15 "
    "de junio de 2026, de https://www.dane.gov.co/index.php/estadisticas-por-tema/"
    "demografia-y-poblacion/proyecciones-de-poblacion/proyecciones-de-poblacion-bogota"
)
st.markdown(
    "- **Ciudad Gótica:** Datos sintéticos generados por inteligencia artificial. "
    "(2026). *Ciudad Gótica: proyección de casos extremos de estrés hídrico* "
    "[Conjunto de datos sintético generado por IA]. Producido mediante un modelo de "
    "lenguaje de gran escala a partir de parámetros extremos; datos ficticios, sin "
    "correspondencia con una población real."
)