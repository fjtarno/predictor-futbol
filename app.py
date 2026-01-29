import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from pathlib import Path

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Analizador Pro - StatsBomb Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚽"
)

# Configuración de rutas multiplataforma
RUTA_BASE = Path(__file__).parent.absolute()
REGISTRO_FILE = RUTA_BASE / "registro_pronosticos.xlsx"

# ==================== FUNCIONES AUXILIARES ====================

def get_pred_str(val, umbral_propension=0.15):
    """
    Calcula la línea .5 más cercana y determina +/- según PROPENSIÓN.
    
    Args:
        val (float): Valor numérico predicho
        umbral_propension (float): Distancia mínima a la línea para considerar tendencia clara
    
    Returns:
        str: Predicción formateada (ej: "+ de 9.5 (9.75)" o "- de 9.5 (9.25)")
    
    Lógica mejorada:
    - Si val está MUY cerca de la línea (±umbral): "≈ [linea]"
    - Si val > linea + umbral: "+ de [linea]"  
    - Si val < linea - umbral: "- de [linea]"
    - Muestra siempre el valor exacto entre paréntesis
    """
    if pd.isna(val) or val <= 0:
        return "0"
    
    linea = int(val) + 0.5
    diferencia = val - linea
    
    # Propensión clara hacia arriba
    if diferencia > umbral_propension:
        return f"+ de {linea} ({val:.2f})"
    # Propensión clara hacia abajo
    elif diferencia < -umbral_propension:
        return f"- de {linea} ({val:.2f})"
    # Zona gris (muy cercano a la línea)
    else:
        return f"≈ {linea} ({val:.2f})"


def cargar_archivo(ruta, tipo='csv'):
    """
    Carga un archivo CSV o Excel con manejo robusto de errores.
    
    Args:
        ruta (Path): Ruta del archivo
        tipo (str): 'csv' o 'excel'
    
    Returns:
        pd.DataFrame o None: DataFrame cargado o None si hay error
    """
    try:
        if tipo == 'csv':
            # Intentar múltiples encodings
            for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(ruta, sep=None, engine='python', encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"No se pudo decodificar {ruta.name}")
        else:
            df = pd.read_excel(ruta)
            return df
    except FileNotFoundError:
        st.error(f"❌ Archivo no encontrado: {ruta.name}")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar {ruta.name}: {str(e)}")
        return None


def validar_dataframe(df, columnas_requeridas, nombre_archivo):
    """
    Valida que un DataFrame tenga las columnas necesarias.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        columnas_requeridas (list): Lista de columnas obligatorias
        nombre_archivo (str): Nombre del archivo para mensajes de error
    
    Returns:
        bool: True si es válido, False en caso contrario
    """
    if df is None:
        return False
    
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        st.error(f"⚠️ {nombre_archivo} - Columnas faltantes: {', '.join(columnas_faltantes)}")
        return False
    
    return True


def crear_grafico_comparacion(equipo_local, equipo_visitante, 
                               exp_c_loc, exp_c_vis, 
                               final_t_loc, final_t_vis):
    """
    Crea gráficos de barras comparativos entre equipos.
    
    Returns:
        tuple: (figura_corners, figura_tarjetas)
    """
    # Gráfico de córners
    fig_corners = go.Figure(data=[
        go.Bar(name='Córners', x=[equipo_local, equipo_visitante], 
               y=[exp_c_loc, exp_c_vis],
               marker_color=['#2ecc71', '#e74c3c'],
               text=[f"{exp_c_loc:.2f}", f"{exp_c_vis:.2f}"],
               textposition='auto')
    ])
    fig_corners.update_layout(
        title="Proyección de Córners por Equipo",
        yaxis_title="Córners Esperados",
        template="plotly_white",
        height=300,
        showlegend=False
    )
    
    # Gráfico de tarjetas
    fig_tarjetas = go.Figure(data=[
        go.Bar(name='Tarjetas', x=[equipo_local, equipo_visitante], 
               y=[final_t_loc, final_t_vis],
               marker_color=['#f39c12', '#9b59b6'],
               text=[f"{final_t_loc:.2f}", f"{final_t_vis:.2f}"],
               textposition='auto')
    ])
    fig_tarjetas.update_layout(
        title="Proyección de Tarjetas por Equipo",
        yaxis_title="Tarjetas Esperadas",
        template="plotly_white",
        height=300,
        showlegend=False
    )
    
    return fig_corners, fig_tarjetas


def crear_grafico_tendencia(df_registro, columna, titulo):
    """
    Crea un gráfico de línea con la evolución temporal de predicciones.
    
    Args:
        df_registro (pd.DataFrame): Registro histórico
        columna (str): Columna a graficar
        titulo (str): Título del gráfico
    
    Returns:
        plotly.graph_objects.Figure: Figura de Plotly
    """
    if df_registro is None or len(df_registro) == 0:
        return None
    
    # Extraer valores numéricos
    def extraer_valor(pred_str):
        if pd.isna(pred_str) or pred_str == "0":
            return 0
        try:
            # Extraer el número entre paréntesis
            return float(str(pred_str).split('(')[1].split(')')[0])
        except:
            return 0
    
    df_plot = df_registro.copy()
    df_plot['valor'] = df_plot[columna].apply(extraer_valor)
    df_plot['index'] = range(len(df_plot))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['index'],
        y=df_plot['valor'],
        mode='lines+markers',
        name=columna,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Predicción #",
        yaxis_title="Valor",
        template="plotly_white",
        height=300
    )
    
    return fig


# ==================== STATSBOMB INTEGRATION ====================

@st.cache_data(ttl=3600)
def cargar_statsbomb_competitions():
    """
    Intenta cargar competiciones disponibles en StatsBomb Open Data.
    Usa caché para evitar llamadas repetidas.
    
    Returns:
        pd.DataFrame o None: DataFrame con competiciones disponibles
    """
    try:
        import statsbombpy as sb
        comps = sb.competitions()
        return comps
    except ImportError:
        st.warning("⚠️ statsbombpy no instalado. Instala con: pip install statsbombpy")
        return None
    except Exception as e:
        st.warning(f"⚠️ No se pudo conectar a StatsBomb: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def obtener_estadisticas_equipo_statsbomb(nombre_equipo, competition_id=11, season_id=90):
    """
    Obtiene estadísticas avanzadas de un equipo desde StatsBomb.
    
    Args:
        nombre_equipo (str): Nombre del equipo
        competition_id (int): ID de la competición (11 = La Liga)
        season_id (int): ID de la temporada (90 = 2020/21)
    
    Returns:
        dict: Diccionario con estadísticas del equipo o None
    """
    try:
        import statsbombpy as sb
        
        # Obtener eventos de la competición
        events = sb.competition_events(
            country="Spain",
            division="La Liga",
            season="2020/2021"
        )
        
        # Filtrar eventos del equipo
        eventos_equipo = events[events['team'] == nombre_equipo]
        
        if len(eventos_equipo) == 0:
            return None
        
        # Calcular estadísticas relevantes
        stats = {
            'presiones_por_partido': len(eventos_equipo[eventos_equipo['type'] == 'Pressure']) / eventos_equipo['match_id'].nunique(),
            'duelos_ganados_pct': len(eventos_equipo[(eventos_equipo['type'] == 'Duel') & (eventos_equipo['duel_outcome'] == 'Won')]) / len(eventos_equipo[eventos_equipo['type'] == 'Duel']) * 100 if len(eventos_equipo[eventos_equipo['type'] == 'Duel']) > 0 else 0,
            'pases_completados_pct': len(eventos_equipo[(eventos_equipo['type'] == 'Pass') & (eventos_equipo['pass_outcome'].isna())]) / len(eventos_equipo[eventos_equipo['type'] == 'Pass']) * 100 if len(eventos_equipo[eventos_equipo['type'] == 'Pass']) > 0 else 0,
            'intercepciones_por_partido': len(eventos_equipo[eventos_equipo['type'] == 'Interception']) / eventos_equipo['match_id'].nunique(),
            'faltas_por_partido': len(eventos_equipo[eventos_equipo['type'] == 'Foul Committed']) / eventos_equipo['match_id'].nunique(),
        }
        
        return stats
        
    except ImportError:
        return None
    except Exception as e:
        st.warning(f"⚠️ Error al obtener stats de {nombre_equipo}: {str(e)}")
        return None


def calcular_factor_intensidad_statsbomb(stats_local, stats_visitante):
    """
    Calcula un factor de ajuste basado en estadísticas de intensidad del juego.
    Mayor intensidad = más córners y tarjetas esperados.
    
    Args:
        stats_local (dict): Stats del equipo local
        stats_visitante (dict): Stats del equipo visitante
    
    Returns:
        dict: Factores de ajuste para córners y tarjetas
    """
    if not stats_local or not stats_visitante:
        return {'corners': 1.0, 'tarjetas': 1.0}
    
    # Factor córners basado en presión e intercepciones
    intensidad_presion = (
        stats_local.get('presiones_por_partido', 0) + 
        stats_visitante.get('presiones_por_partido', 0)
    ) / 2
    
    # Normalizar (asumiendo media de ~150 presiones/partido)
    factor_corners = 1.0 + ((intensidad_presion - 150) / 150) * 0.1
    factor_corners = max(0.9, min(1.1, factor_corners))  # Limitar entre 0.9 y 1.1
    
    # Factor tarjetas basado en faltas
    faltas_totales = (
        stats_local.get('faltas_por_partido', 0) + 
        stats_visitante.get('faltas_por_partido', 0)
    )
    
    # Normalizar (asumiendo media de ~25 faltas/partido)
    factor_tarjetas = 1.0 + ((faltas_totales - 25) / 25) * 0.15
    factor_tarjetas = max(0.85, min(1.15, factor_tarjetas))  # Limitar entre 0.85 y 1.15
    
    return {
        'corners': factor_corners,
        'tarjetas': factor_tarjetas,
        'intensidad_presion': intensidad_presion,
        'faltas_totales': faltas_totales
    }


# ==================== INICIALIZACIÓN ====================

if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {}

if 'peso_reciente' not in st.session_state:
    st.session_state['peso_reciente'] = 0.7

if 'usar_statsbomb' not in st.session_state:
    st.session_state['usar_statsbomb'] = False

if 'umbral_propension' not in st.session_state:
    st.session_state['umbral_propension'] = 0.15

# ==================== INTERFAZ PRINCIPAL ====================

st.title("⚽ Sistema de Análisis Estadístico Profesional")
st.caption("Enhanced con StatsBomb Open Data Integration")
st.markdown("---")

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración Avanzada")
    
    st.subheader("🎯 Modelo de Predicción")
    st.session_state['peso_reciente'] = st.slider(
        "Peso datos recientes",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Mayor valor = más importancia a partidos recientes"
    )
    
    st.session_state['umbral_propension'] = st.slider(
        "Umbral de propensión (+/-)",
        min_value=0.05,
        max_value=0.30,
        value=0.15,
        step=0.05,
        help="Distancia mínima a la línea .5 para considerar tendencia clara hacia +/- de la línea"
    )
    
    st.markdown("---")
    
    st.subheader("🌐 StatsBomb Integration")
    
    # Comprobar si statsbombpy está disponible
    try:
        import statsbombpy as sb
        statsbomb_disponible = True
        st.success("✅ statsbombpy instalado")
    except ImportError:
        statsbomb_disponible = False
        st.error("❌ statsbombpy no instalado")
        st.code("pip install statsbombpy", language="bash")
    
    if statsbomb_disponible:
        st.session_state['usar_statsbomb'] = st.checkbox(
            "Enriquecer con StatsBomb",
            value=False,
            help="Añade factores de intensidad basados en datos de StatsBomb"
        )
        
        if st.session_state['usar_statsbomb']:
            st.info("💡 Los cálculos incluirán factores de presión, duelos e intensidad del juego")
    
    st.markdown("---")
    st.caption("💡 **Tip:** Ajusta los parámetros según tu estrategia de análisis")

# Definición de archivos objetivo
archivos_objetivo = {
    "Promedio Corners": "1.ESP_promedio corners",
    "Corners Local": "2.ESP_corners local",
    "Corners Visitante": "3.ESP_corners visitante",
    "Tarjetas Casa": "4.ESP_Tarjetas_casa",
    "Tarjetas Fuera": "5.ESP_Tarjetas_fuera",
    "Árbitros": "6.ESP_aRBITRO"
}

# Columnas requeridas por tipo de archivo
columnas_requeridas = {
    "Corners Local": ["Team", "CFH", "CAH"],
    "Corners Visitante": ["Team", "CFA", "CAA"],
    "Tarjetas Casa": ["Team", "YFH", "YAH"],
    "Tarjetas Fuera": ["Team", "YFA", "YAA"],
    "Árbitros": ["Árbitro", "A/P"]
}

# ==================== PESTAÑAS ====================

tab_carga, tab_seleccion, tab_analisis, tab_predicciones, tab_backtesting, tab_statsbomb = st.tabs([
    "📥 Gestión de Datos", 
    "⚽ Configuración", 
    "📈 Análisis Comparativo", 
    "🎯 Resultados y Registro",
    "📊 Backtesting & Métricas",
    "🌐 StatsBomb Insights"
])

# ==================== PESTAÑA 1: CARGA DE DATOS ====================

with tab_carga:
    st.header("🔄 Sincronización de Base de Datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"📁 **Ruta actual:** `{RUTA_BASE}`")
    
    with col2:
        if st.button("🔄 Sincronizar Carpeta Local", type="primary"):
            with st.spinner("Cargando archivos..."):
                errores = []
                exitos = []
                
                try:
                    archivos_en_carpeta = list(RUTA_BASE.glob("*"))
                    
                    for etiqueta, nombre_buscado in archivos_objetivo.items():
                        archivo_encontrado = False
                        
                        for archivo_path in archivos_en_carpeta:
                            if archivo_path.name.startswith(nombre_buscado):
                                # Determinar tipo de archivo
                                extension = archivo_path.suffix.lower()
                                tipo = 'csv' if extension == '.csv' else 'excel'
                                
                                # Cargar archivo
                                df = cargar_archivo(archivo_path, tipo)
                                
                                if df is not None:
                                    # Validar columnas si es necesario
                                    if etiqueta in columnas_requeridas:
                                        if validar_dataframe(df, columnas_requeridas[etiqueta], etiqueta):
                                            st.session_state['dfs'][etiqueta] = df
                                            exitos.append(f"✅ {etiqueta}: {len(df)} registros")
                                            archivo_encontrado = True
                                        else:
                                            errores.append(f"❌ {etiqueta}: validación fallida")
                                    else:
                                        st.session_state['dfs'][etiqueta] = df
                                        exitos.append(f"✅ {etiqueta}: {len(df)} registros")
                                        archivo_encontrado = True
                                    break
                        
                        if not archivo_encontrado:
                            errores.append(f"⚠️ {etiqueta}: archivo no encontrado")
                
                except Exception as e:
                    st.error(f"❌ Error durante la sincronización: {str(e)}")
                
                # Mostrar resultados
                if exitos:
                    for msg in exitos:
                        st.success(msg)
                
                if errores:
                    st.warning("**Advertencias:**")
                    for msg in errores:
                        st.write(msg)
    
    st.markdown("---")
    
    # Estado de la carga
    st.subheader("📊 Estado de los Datos")
    
    if len(st.session_state['dfs']) > 0:
        col_status = st.columns(3)
        
        for idx, (etiqueta, nombre) in enumerate(archivos_objetivo.items()):
            with col_status[idx % 3]:
                if etiqueta in st.session_state['dfs']:
                    st.success(f"✅ {etiqueta}")
                    st.caption(f"{len(st.session_state['dfs'][etiqueta])} registros")
                else:
                    st.error(f"❌ {etiqueta}")
    else:
        st.info("ℹ️ No hay datos cargados. Haz clic en 'Sincronizar Carpeta Local'.")

# ==================== PROCESAMIENTO PRINCIPAL ====================

if len(st.session_state['dfs']) >= 6:
    dfs = st.session_state['dfs']
    
    # ==================== PESTAÑA 2: SELECCIÓN ====================
    
    with tab_seleccion:
        st.header("⚽ Configuración del Partido")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            equipos_local = sorted(dfs["Corners Local"]["Team"].unique())
            equipo_local = st.selectbox(
                "🏠 Equipo Local",
                equipos_local,
                help="Selecciona el equipo que juega en casa"
            )
        
        with col2:
            equipos_visitante = sorted(dfs["Corners Visitante"]["Team"].unique())
            equipo_visitante = st.selectbox(
                "✈️ Equipo Visitante",
                equipos_visitante,
                help="Selecciona el equipo que juega fuera"
            )
        
        with col3:
            arbitros = sorted(dfs["Árbitros"]["Árbitro"].unique())
            arbitro_sel = st.selectbox(
                "🎽 Árbitro",
                arbitros,
                help="Selecciona el árbitro del partido"
            )
        
        st.markdown("---")
        
        # Validar que los equipos no sean iguales
        if equipo_local == equipo_visitante:
            st.warning("⚠️ El equipo local y visitante no pueden ser el mismo")
            st.stop()
        
        # Información rápida de los equipos seleccionados
        st.subheader("📋 Resumen de Selección")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.info(f"**Local:** {equipo_local}")
        with col_info2:
            st.info(f"**Visitante:** {equipo_visitante}")
        with col_info3:
            st.info(f"**Árbitro:** {arbitro_sel}")
    
    # ==================== CÁLCULOS BASE ====================
    
    try:
        # Filtrar datos de cada equipo
        filtro_local_corners = dfs["Corners Local"]["Team"] == equipo_local
        filtro_visitante_corners = dfs["Corners Visitante"]["Team"] == equipo_visitante
        filtro_local_tarjetas = dfs["Tarjetas Casa"]["Team"] == equipo_local
        filtro_visitante_tarjetas = dfs["Tarjetas Fuera"]["Team"] == equipo_visitante
        filtro_arbitro = dfs["Árbitros"]["Árbitro"] == arbitro_sel
        
        # Verificar que existan los datos
        if not filtro_local_corners.any():
            st.error(f"❌ No hay datos de córners para {equipo_local} como local")
            st.stop()
        if not filtro_visitante_corners.any():
            st.error(f"❌ No hay datos de córners para {equipo_visitante} como visitante")
            st.stop()
        if not filtro_local_tarjetas.any():
            st.error(f"❌ No hay datos de tarjetas para {equipo_local} como local")
            st.stop()
        if not filtro_visitante_tarjetas.any():
            st.error(f"❌ No hay datos de tarjetas para {equipo_visitante} como visitante")
            st.stop()
        if not filtro_arbitro.any():
            st.error(f"❌ No hay datos para el árbitro {arbitro_sel}")
            st.stop()
        
        # Obtener datos
        dcl = dfs["Corners Local"][filtro_local_corners].iloc[0]
        dcv = dfs["Corners Visitante"][filtro_visitante_corners].iloc[0]
        dtl = dfs["Tarjetas Casa"][filtro_local_tarjetas].iloc[0]
        dtv = dfs["Tarjetas Fuera"][filtro_visitante_tarjetas].iloc[0]
        dar = dfs["Árbitros"][filtro_arbitro].iloc[0]
        
        # Cálculos Córners BASE
        exp_c_loc_base = (dcl['CFH'] + dcv['CAA']) / 2
        exp_c_vis_base = (dcv['CFA'] + dcl['CAH']) / 2
        
        # Cálculos Tarjetas BASE + Factor Árbitro
        media_amarillas_liga = dfs["Árbitros"]["A/P"].mean()
        factor_arbitro = dar['A/P'] / media_amarillas_liga if media_amarillas_liga > 0 else 1
        
        final_t_loc_base = ((dtl['YFH'] + dtv['YAA']) / 2) * factor_arbitro
        final_t_vis_base = ((dtv['YFA'] + dtl['YAH']) / 2) * factor_arbitro
        
        # ==================== ENRIQUECIMIENTO CON STATSBOMB ====================
        
        factores_statsbomb = {'corners': 1.0, 'tarjetas': 1.0}
        stats_sb_local = None
        stats_sb_visitante = None
        
        if st.session_state.get('usar_statsbomb', False):
            with st.spinner("🌐 Cargando datos de StatsBomb..."):
                stats_sb_local = obtener_estadisticas_equipo_statsbomb(equipo_local)
                stats_sb_visitante = obtener_estadisticas_equipo_statsbomb(equipo_visitante)
                
                if stats_sb_local and stats_sb_visitante:
                    factores_statsbomb = calcular_factor_intensidad_statsbomb(
                        stats_sb_local, 
                        stats_sb_visitante
                    )
        
        # Aplicar factores de StatsBomb
        exp_c_loc = exp_c_loc_base * factores_statsbomb['corners']
        exp_c_vis = exp_c_vis_base * factores_statsbomb['corners']
        total_c = exp_c_loc + exp_c_vis
        
        final_t_loc = final_t_loc_base * factores_statsbomb['tarjetas']
        final_t_vis = final_t_vis_base * factores_statsbomb['tarjetas']
        total_t = final_t_loc + final_t_vis
        
    except KeyError as e:
        st.error(f"❌ Error: columna faltante en los datos - {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error en los cálculos: {str(e)}")
        st.stop()
    
    # ==================== PESTAÑA 3: ANÁLISIS COMPARATIVO ====================
    
    with tab_analisis:
        st.header(f"🔍 Análisis de Emparejamiento: {equipo_local} vs {equipo_visitante}")
        
        # Mostrar si se usó StatsBomb
        if st.session_state.get('usar_statsbomb', False) and stats_sb_local and stats_sb_visitante:
            st.success(f"✅ Análisis enriquecido con StatsBomb (Factor Córners: {factores_statsbomb['corners']:.2f}x, Factor Tarjetas: {factores_statsbomb['tarjetas']:.2f}x)")
        
        # Gráficos comparativos
        fig_c, fig_t = crear_grafico_comparacion(
            equipo_local, equipo_visitante,
            exp_c_loc, exp_c_vis,
            final_t_loc, final_t_vis
        )
        
        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            st.plotly_chart(fig_c, use_container_width=True)
        with col_graph2:
            st.plotly_chart(fig_t, use_container_width=True)
        
        st.markdown("---")
        
        # Análisis de Córners
        st.subheader("⚖️ Estimaciones de Córners (Ataque vs Concesión)")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.metric(
                f"🏠 {equipo_local}",
                f"{exp_c_loc:.2f} córners",
                help=f"Saca {dcl['CFH']:.2f} y el rival permite {dcv['CAA']:.2f}"
            )
            with st.expander("📊 Desglose Local"):
                st.write(f"**Córners a favor (casa):** {dcl['CFH']:.2f}")
                st.write(f"**Córners en contra permitidos (casa):** {dcl['CAH']:.2f}")
                st.write(f"**Cálculo base:** {exp_c_loc_base:.2f}")
                if factores_statsbomb['corners'] != 1.0:
                    st.write(f"**Factor StatsBomb:** {factores_statsbomb['corners']:.2f}x")
        
        with col_c2:
            st.metric(
                f"✈️ {equipo_visitante}",
                f"{exp_c_vis:.2f} córners",
                help=f"Saca {dcv['CFA']:.2f} y el rival permite {dcl['CAH']:.2f}"
            )
            with st.expander("📊 Desglose Visitante"):
                st.write(f"**Córners a favor (fuera):** {dcv['CFA']:.2f}")
                st.write(f"**Córners en contra permitidos (fuera):** {dcv['CAA']:.2f}")
                st.write(f"**Cálculo base:** {exp_c_vis_base:.2f}")
                if factores_statsbomb['corners'] != 1.0:
                    st.write(f"**Factor StatsBomb:** {factores_statsbomb['corners']:.2f}x")
        
        st.success(f"**📍 TOTAL PROYECTADO: {total_c:.2f} Córners**")
        
        st.markdown("---")
        
        # Análisis de Tarjetas
        st.subheader("⚖️ Estimaciones de Tarjetas (Ajustadas por Árbitro)")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            st.metric("📊 Media Liga", f"{media_amarillas_liga:.2f} tarjetas")
        
        with col_t2:
            st.metric(f"🎽 {arbitro_sel}", f"{dar['A/P']:.2f} tarjetas")
        
        with col_t3:
            impacto = (factor_arbitro - 1) * 100
            st.metric(
                "Impacto Árbitro",
                f"{impacto:+.1f}%",
                delta=f"Factor: {factor_arbitro:.2f}"
            )
        
        if factor_arbitro > 1:
            st.warning(f"⚠️ El árbitro **{arbitro_sel}** INCREMENTA la proyección de tarjetas en un **{abs(1-factor_arbitro)*100:.1f}%** respecto a la media.")
        elif factor_arbitro < 1:
            st.info(f"ℹ️ El árbitro **{arbitro_sel}** REDUCE la proyección de tarjetas en un **{abs(1-factor_arbitro)*100:.1f}%** respecto a la media.")
        else:
            st.success(f"✅ El árbitro **{arbitro_sel}** está en línea con la media de la liga.")
        
        st.markdown("---")
        
        # Proyecciones finales de tarjetas
        col_tf1, col_tf2 = st.columns(2)
        
        with col_tf1:
            st.metric(f"🟨 {equipo_local}", f"{final_t_loc:.2f} tarjetas")
            with st.expander("📊 Desglose Tarjetas Local"):
                st.write(f"**A favor (Recibe):** {dtl['YFH']:.2f} tarjetas/partido")
                st.write(f"**En contra (Provoca):** {dtl['YAH']:.2f} tarjetas/partido")
                st.write(f"**Base sin árbitro:** {(dtl['YFH'] + dtv['YAA']) / 2:.2f}")
                st.write(f"**Factor árbitro aplicado:** {factor_arbitro:.2f}")
                if factores_statsbomb['tarjetas'] != 1.0:
                    st.write(f"**Factor StatsBomb:** {factores_statsbomb['tarjetas']:.2f}x")
        
        with col_tf2:
            st.metric(f"🟥 {equipo_visitante}", f"{final_t_vis:.2f} tarjetas")
            with st.expander("📊 Desglose Tarjetas Visitante"):
                st.write(f"**A favor (Recibe):** {dtv['YFA']:.2f} tarjetas/partido")
                st.write(f"**En contra (Provoca):** {dtv['YAA']:.2f} tarjetas/partido")
                st.write(f"**Base sin árbitro:** {(dtv['YFA'] + dtl['YAH']) / 2:.2f}")
                st.write(f"**Factor árbitro aplicado:** {factor_arbitro:.2f}")
                if factores_statsbomb['tarjetas'] != 1.0:
                    st.write(f"**Factor StatsBomb:** {factores_statsbomb['tarjetas']:.2f}x")
        
        st.success(f"**📍 TOTAL PROYECTADO: {total_t:.2f} Tarjetas**")
        
        st.markdown("---")
        
        # Informe de Probabilidades
        st.subheader("🔎 Informe de Probabilidades")
        
        equipo_fragil = equipo_visitante if dcv['CAA'] > dcl['CAH'] else equipo_local
        max_corners_concedidos = max(dcv['CAA'], dcl['CAH'])
        
        st.info(f"""
        **Análisis de Córners:**
        En este encuentro, el volumen de córners se ve influenciado por la fragilidad del **{equipo_fragil}**, 
        que concede **{max_corners_concedidos:.2f}** córners por partido.
        """)
        
        if factor_arbitro > 1.10:
            st.warning(f"""
            **⚠️ ALERTA ÁRBITRO:**
            El colegiado **{arbitro_sel}** muestra un perfil estricto con **{dar['A/P']:.2f}** tarjetas/partido,
            lo que dispara significativamente la probabilidad de tarjetas en este encuentro.
            """)
        elif factor_arbitro < 0.90:
            st.success(f"""
            **✅ ÁRBITRO PERMISIVO:**
            El colegiado **{arbitro_sel}** tiene un perfil permisivo con **{dar['A/P']:.2f}** tarjetas/partido,
            lo que reduce la expectativa de tarjetas.
            """)
    
    # ==================== PESTAÑA 4: RESULTADOS Y REGISTRO ====================
    
    with tab_predicciones:
        st.header("🎯 Dashboard de Resultados Finales")
        
        # Explicación del nuevo sistema +/-
        with st.expander("ℹ️ ¿Cómo interpretar los signos + / - ?"):
            st.markdown(f"""
            **Sistema de Propensión Mejorado:**
            
            - **+ de X.5**: La predicción tiene propensión CLARA hacia MÁS de la línea  
              _(valor > línea + {st.session_state['umbral_propension']})_
            
            - **- de X.5**: La predicción tiene propensión CLARA hacia MENOS de la línea  
              _(valor < línea - {st.session_state['umbral_propension']})_
            
            - **≈ X.5**: La predicción está muy CERCA de la línea (zona gris)  
              _(distancia a la línea < {st.session_state['umbral_propension']})_
            
            **Ejemplo:**
            - 9.75 córners → **"+ de 9.5"** (propensión clara hacia más de 9.5)
            - 9.25 córners → **"- de 9.5"** (propensión clara hacia menos de 9.5)
            - 9.45 córners → **"≈ 9.5"** (muy cerca, sin tendencia clara)
            
            _Ajusta el umbral en la barra lateral según tu agresividad de apuesta._
            """)
        
        # Sección Córners formateada con nuevo sistema
        st.subheader("🚩 Córners")
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            pred_c_total = get_pred_str(total_c, st.session_state['umbral_propension'])
            st.metric(
                "🎯 Córners TOTAL",
                pred_c_total,
                help=f"Valor exacto: {total_c:.2f}"
            )
        
        with rc2:
            pred_c_loc = get_pred_str(exp_c_loc, st.session_state['umbral_propension'])
            st.metric(
                f"🏠 {equipo_local}",
                pred_c_loc,
                help=f"Valor exacto: {exp_c_loc:.2f}"
            )
        
        with rc3:
            pred_c_vis = get_pred_str(exp_c_vis, st.session_state['umbral_propension'])
            st.metric(
                f"✈️ {equipo_visitante}",
                pred_c_vis,
                help=f"Valor exacto: {exp_c_vis:.2f}"
            )
        
        st.markdown("---")
        
        # Sección Tarjetas formateada con nuevo sistema
        st.subheader("🟨 Tarjetas")
        rt1, rt2, rt3 = st.columns(3)
        
        with rt1:
            pred_t_total = get_pred_str(total_t, st.session_state['umbral_propension'])
            st.metric(
                "🎯 Tarjetas TOTAL",
                pred_t_total,
                help=f"Valor exacto: {total_t:.2f}"
            )
        
        with rt2:
            pred_t_loc = get_pred_str(final_t_loc, st.session_state['umbral_propension'])
            st.metric(
                f"🏠 {equipo_local}",
                pred_t_loc,
                help=f"Valor exacto: {final_t_loc:.2f}"
            )
        
        with rt3:
            pred_t_vis = get_pred_str(final_t_vis, st.session_state['umbral_propension'])
            st.metric(
                f"✈️ {equipo_visitante}",
                pred_t_vis,
                help=f"Valor exacto: {final_t_vis:.2f}"
            )
        
        st.markdown("---")
        
        # Registro Histórico
        st.subheader("💾 Registro Histórico")
        
        col_export1, col_export2 = st.columns([2, 1])
        
        with col_export1:
            st.info("💡 Exporta esta predicción al registro histórico para análisis posterior.")
        
        with col_export2:
            if st.button("📥 Exportar a Registro Excel", type="primary"):
                # Crear nuevo registro con formato mejorado
                nuevo_dato = {
                    "Fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Local": equipo_local,
                    "Visitante": equipo_visitante,
                    "Árbitro": arbitro_sel,
                    "Corn_Tot": pred_c_total,
                    "Corn_Loc": pred_c_loc,
                    "Corn_Vis": pred_c_vis,
                    "Tarj_Tot": pred_t_total,
                    "Tarj_Loc": pred_t_loc,
                    "Tarj_Vis": pred_t_vis,
                    # Valores numéricos para análisis posterior
                    "Corn_Tot_Num": round(total_c, 2),
                    "Corn_Loc_Num": round(exp_c_loc, 2),
                    "Corn_Vis_Num": round(exp_c_vis, 2),
                    "Tarj_Tot_Num": round(total_t, 2),
                    "Tarj_Loc_Num": round(final_t_loc, 2),
                    "Tarj_Vis_Num": round(final_t_vis, 2),
                    "Factor_Arbitro": round(factor_arbitro, 2),
                    "Factor_StatsBomb_Corners": round(factores_statsbomb['corners'], 2),
                    "Factor_StatsBomb_Tarjetas": round(factores_statsbomb['tarjetas'], 2),
                    "Umbral_Propension": st.session_state['umbral_propension'],
                    # Campos para resultados reales (a completar manualmente)
                    "Corn_Tot_Real": None,
                    "Tarj_Tot_Real": None
                }
                
                try:
                    # Cargar o crear DataFrame
                    if REGISTRO_FILE.exists():
                        df_actual = pd.read_excel(REGISTRO_FILE)
                        df_final = pd.concat([df_actual, pd.DataFrame([nuevo_dato])], ignore_index=True)
                    else:
                        df_final = pd.DataFrame([nuevo_dato])
                    
                    # Guardar
                    df_final.to_excel(REGISTRO_FILE, index=False)
                    
                    st.success(f"✅ Predicción exportada correctamente: {len(df_final)} registros totales")
                    st.balloons()
                    
                except PermissionError:
                    st.error("❌ Error: El archivo Excel está abierto. Ciérralo e intenta nuevamente.")
                except Exception as e:
                    st.error(f"❌ Error al exportar: {str(e)}")
        
        st.markdown("---")
        
        # Mostrar últimas predicciones
        if REGISTRO_FILE.exists():
            try:
                df_registro = pd.read_excel(REGISTRO_FILE)
                
                if len(df_registro) > 0:
                    st.subheader("📜 Últimas 5 Predicciones")
                    
                    # Seleccionar columnas relevantes para mostrar
                    columnas_mostrar = ["Fecha", "Local", "Visitante", "Árbitro", 
                                       "Corn_Tot", "Tarj_Tot"]
                    
                    df_mostrar = df_registro[columnas_mostrar].tail(5).iloc[::-1]
                    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)
                    
                    # Botón para ver registro completo
                    with st.expander("📋 Ver Registro Completo"):
                        st.dataframe(df_registro, use_container_width=True)
            
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar el registro: {str(e)}")
    
    # ==================== PESTAÑA 5: BACKTESTING ====================
    
    with tab_backtesting:
        st.header("📊 Backtesting & Análisis de Rendimiento")
        
        if REGISTRO_FILE.exists():
            try:
                df_registro = pd.read_excel(REGISTRO_FILE)
                
                if len(df_registro) > 0:
                    # Métricas generales
                    st.subheader("📈 Estadísticas Generales")
                    
                    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                    
                    with col_met1:
                        st.metric("Total Predicciones", len(df_registro))
                    
                    with col_met2:
                        if 'Corn_Tot_Num' in df_registro.columns:
                            promedio_corners = df_registro['Corn_Tot_Num'].mean()
                            st.metric("Promedio Córners", f"{promedio_corners:.2f}")
                        else:
                            st.metric("Promedio Córners", "N/A")
                    
                    with col_met3:
                        if 'Tarj_Tot_Num' in df_registro.columns:
                            promedio_tarjetas = df_registro['Tarj_Tot_Num'].mean()
                            st.metric("Promedio Tarjetas", f"{promedio_tarjetas:.2f}")
                        else:
                            st.metric("Promedio Tarjetas", "N/A")
                    
                    with col_met4:
                        if 'Factor_Arbitro' in df_registro.columns:
                            promedio_factor = df_registro['Factor_Arbitro'].mean()
                            st.metric("Factor Árbitro Medio", f"{promedio_factor:.2f}")
                        else:
                            st.metric("Factor Árbitro Medio", "N/A")
                    
                    st.markdown("---")
                    
                    # Gráficos de tendencia
                    st.subheader("📉 Evolución Temporal")
                    
                    col_graf1, col_graf2 = st.columns(2)
                    
                    with col_graf1:
                        if 'Corn_Tot' in df_registro.columns:
                            fig_trend_c = crear_grafico_tendencia(
                                df_registro, 'Corn_Tot', 
                                'Evolución de Predicciones - Córners Totales'
                            )
                            if fig_trend_c:
                                st.plotly_chart(fig_trend_c, use_container_width=True)
                    
                    with col_graf2:
                        if 'Tarj_Tot' in df_registro.columns:
                            fig_trend_t = crear_grafico_tendencia(
                                df_registro, 'Tarj_Tot', 
                                'Evolución de Predicciones - Tarjetas Totales'
                            )
                            if fig_trend_t:
                                st.plotly_chart(fig_trend_t, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Distribuciones
                    st.subheader("📊 Distribuciones")
                    
                    if 'Corn_Tot_Num' in df_registro.columns and 'Tarj_Tot_Num' in df_registro.columns:
                        col_dist1, col_dist2 = st.columns(2)
                        
                        with col_dist1:
                            fig_hist_c = px.histogram(
                                df_registro, 
                                x='Corn_Tot_Num',
                                nbins=20,
                                title='Distribución de Córners Totales',
                                labels={'Corn_Tot_Num': 'Córners'},
                                color_discrete_sequence=['#2ecc71']
                            )
                            fig_hist_c.update_layout(template="plotly_white", height=300)
                            st.plotly_chart(fig_hist_c, use_container_width=True)
                        
                        with col_dist2:
                            fig_hist_t = px.histogram(
                                df_registro, 
                                x='Tarj_Tot_Num',
                                nbins=20,
                                title='Distribución de Tarjetas Totales',
                                labels={'Tarj_Tot_Num': 'Tarjetas'},
                                color_discrete_sequence=['#f39c12']
                            )
                            fig_hist_t.update_layout(template="plotly_white", height=300)
                            st.plotly_chart(fig_hist_t, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Top equipos
                    st.subheader("🏆 Rankings")
                    
                    col_rank1, col_rank2 = st.columns(2)
                    
                    with col_rank1:
                        st.write("**Top 10 Equipos por Apariciones**")
                        equipos_local = df_registro['Local'].value_counts().head(10)
                        equipos_visitante = df_registro['Visitante'].value_counts().head(10)
                        equipos_total = pd.concat([equipos_local, equipos_visitante]).groupby(level=0).sum().sort_values(ascending=False).head(10)
                        
                        st.dataframe(
                            equipos_total.reset_index().rename(columns={'index': 'Equipo', 0: 'Apariciones'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col_rank2:
                        st.write("**Top 10 Árbitros más Frecuentes**")
                        arbitros_freq = df_registro['Árbitro'].value_counts().head(10)
                        
                        st.dataframe(
                            arbitros_freq.reset_index().rename(columns={'index': 'Árbitro', 'count': 'Apariciones'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    st.markdown("---")
                    
                    # Instrucciones para completar resultados reales
                    st.subheader("📝 Análisis de Precisión")
                    
                    st.info("""
                    **💡 Cómo mejorar el análisis:**
                    
                    Para calcular la precisión del modelo, abre el archivo `registro_pronosticos.xlsx` 
                    y completa las columnas:
                    - `Corn_Tot_Real`: Córners totales reales del partido
                    - `Tarj_Tot_Real`: Tarjetas totales reales del partido
                    
                    Una vez completados, esta pestaña mostrará:
                    - Error medio absoluto (MAE)
                    - Tasa de acierto en líneas
                    - Gráficos de comparación predicción vs realidad
                    """)
                    
                    # Verificar si hay datos reales
                    if 'Corn_Tot_Real' in df_registro.columns and 'Tarj_Tot_Real' in df_registro.columns:
                        df_con_reales = df_registro.dropna(subset=['Corn_Tot_Real', 'Tarj_Tot_Real'])
                        
                        if len(df_con_reales) > 0:
                            st.success(f"✅ {len(df_con_reales)} predicciones con resultados reales")
                            
                            # Calcular errores
                            df_con_reales['Error_Corners'] = abs(df_con_reales['Corn_Tot_Num'] - df_con_reales['Corn_Tot_Real'])
                            df_con_reales['Error_Tarjetas'] = abs(df_con_reales['Tarj_Tot_Num'] - df_con_reales['Tarj_Tot_Real'])
                            
                            mae_corners = df_con_reales['Error_Corners'].mean()
                            mae_tarjetas = df_con_reales['Error_Tarjetas'].mean()
                            
                            col_mae1, col_mae2 = st.columns(2)
                            
                            with col_mae1:
                                st.metric("MAE Córners", f"{mae_corners:.2f}")
                            
                            with col_mae2:
                                st.metric("MAE Tarjetas", f"{mae_tarjetas:.2f}")
                            
                            # Gráfico de comparación
                            fig_comparacion = go.Figure()
                            
                            fig_comparacion.add_trace(go.Scatter(
                                x=list(range(len(df_con_reales))),
                                y=df_con_reales['Corn_Tot_Num'],
                                mode='lines+markers',
                                name='Predicción Córners',
                                line=dict(color='blue')
                            ))
                            
                            fig_comparacion.add_trace(go.Scatter(
                                x=list(range(len(df_con_reales))),
                                y=df_con_reales['Corn_Tot_Real'],
                                mode='lines+markers',
                                name='Real Córners',
                                line=dict(color='green', dash='dash')
                            ))
                            
                            fig_comparacion.update_layout(
                                title="Predicción vs Realidad - Córners",
                                xaxis_title="Partido #",
                                yaxis_title="Córners",
                                template="plotly_white",
                                height=400
                            )
                            
                            st.plotly_chart(fig_comparacion, use_container_width=True)
                        else:
                            st.warning("⚠️ No hay predicciones con resultados reales completados")
                
                else:
                    st.info("ℹ️ No hay predicciones en el registro todavía.")
            
            except Exception as e:
                st.error(f"❌ Error al cargar el registro: {str(e)}")
        
        else:
            st.info("ℹ️ No existe archivo de registro. Exporta tu primera predicción para comenzar.")
    
    # ==================== PESTAÑA 6: STATSBOMB INSIGHTS ====================
    
    with tab_statsbomb:
        st.header("🌐 StatsBomb Open Data Insights")
        
        st.markdown("""
        **StatsBomb Open Data** proporciona datos de eventos granulares de partidos profesionales.
        Esta integración enriquece las predicciones con factores de intensidad del juego.
        """)
        
        # Verificar si statsbombpy está instalado
        try:
            import statsbombpy as sb
            st.success("✅ Módulo statsbombpy instalado correctamente")
            
            # Mostrar competiciones disponibles
            st.subheader("📋 Competiciones Disponibles")
            
            with st.spinner("Cargando competiciones de StatsBomb..."):
                comps_sb = cargar_statsbomb_competitions()
                
                if comps_sb is not None:
                    st.write(f"**Total de competiciones:** {len(comps_sb)}")
                    
                    # Filtrar por España / La Liga si existe
                    laliga = comps_sb[comps_sb['competition_name'].str.contains('La Liga', case=False, na=False)]
                    
                    if len(laliga) > 0:
                        st.success(f"✅ La Liga encontrada: {len(laliga)} temporadas disponibles")
                        with st.expander("Ver detalles de La Liga"):
                            st.dataframe(laliga[['competition_name', 'season_name', 'competition_id', 'season_id']], use_container_width=True)
                    else:
                        st.warning("⚠️ No se encontró La Liga en datos abiertos")
                    
                    # Mostrar todas las competiciones
                    with st.expander("📊 Ver todas las competiciones disponibles"):
                        st.dataframe(comps_sb, use_container_width=True)
                else:
                    st.error("❌ No se pudieron cargar las competiciones")
            
            st.markdown("---")
            
            # Estadísticas de los equipos actuales si se usó StatsBomb
            if st.session_state.get('usar_statsbomb', False) and stats_sb_local and stats_sb_visitante:
                st.subheader(f"📊 Estadísticas Avanzadas: {equipo_local} vs {equipo_visitante}")
                
                col_sb1, col_sb2 = st.columns(2)
                
                with col_sb1:
                    st.markdown(f"### 🏠 {equipo_local}")
                    st.metric("Presiones/partido", f"{stats_sb_local.get('presiones_por_partido', 0):.1f}")
                    st.metric("Duelos ganados %", f"{stats_sb_local.get('duelos_ganados_pct', 0):.1f}%")
                    st.metric("Pases completados %", f"{stats_sb_local.get('pases_completados_pct', 0):.1f}%")
                    st.metric("Intercepciones/partido", f"{stats_sb_local.get('intercepciones_por_partido', 0):.1f}")
                    st.metric("Faltas/partido", f"{stats_sb_local.get('faltas_por_partido', 0):.1f}")
                
                with col_sb2:
                    st.markdown(f"### ✈️ {equipo_visitante}")
                    st.metric("Presiones/partido", f"{stats_sb_visitante.get('presiones_por_partido', 0):.1f}")
                    st.metric("Duelos ganados %", f"{stats_sb_visitante.get('duelos_ganados_pct', 0):.1f}%")
                    st.metric("Pases completados %", f"{stats_sb_visitante.get('pases_completados_pct', 0):.1f}%")
                    st.metric("Intercepciones/partido", f"{stats_sb_visitante.get('intercepciones_por_partido', 0):.1f}")
                    st.metric("Faltas/partido", f"{stats_sb_visitante.get('faltas_por_partido', 0):.1f}")
                
                st.markdown("---")
                
                st.subheader("🎯 Factores de Ajuste Calculados")
                
                col_factor1, col_factor2 = st.columns(2)
                
                with col_factor1:
                    st.metric(
                        "Factor Córners",
                        f"{factores_statsbomb['corners']:.2f}x",
                        help="Basado en intensidad de presión del juego"
                    )
                    if 'intensidad_presion' in factores_statsbomb:
                        st.caption(f"Intensidad de presión: {factores_statsbomb['intensidad_presion']:.1f} presiones/partido")
                
                with col_factor2:
                    st.metric(
                        "Factor Tarjetas",
                        f"{factores_statsbomb['tarjetas']:.2f}x",
                        help="Basado en faltas cometidas por partido"
                    )
                    if 'faltas_totales' in factores_statsbomb:
                        st.caption(f"Faltas totales: {factores_statsbomb['faltas_totales']:.1f} faltas/partido")
            
            elif not st.session_state.get('usar_statsbomb', False):
                st.info("ℹ️ Activa 'Enriquecer con StatsBomb' en la barra lateral para ver estadísticas avanzadas de los equipos seleccionados.")
            else:
                st.warning("⚠️ No se encontraron datos de StatsBomb para los equipos seleccionados. Los equipos deben estar en La Liga 2020/21 (datos abiertos).")
            
            st.markdown("---")
            
            # Información adicional
            st.subheader("📚 Recursos")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("""
                **Enlaces Útiles:**
                - [StatsBomb Open Data GitHub](https://github.com/statsbomb/open-data)
                - [statsbombpy Documentation](https://github.com/statsbomb/statsbombpy)
                - [StatsBomb Website](https://statsbomb.com)
                """)
            
            with col_rec2:
                st.markdown("""
                **Cómo usar:**
                1. Activa "Enriquecer con StatsBomb" en sidebar
                2. Las predicciones incluirán factores de intensidad
                3. Los factores ajustan córners y tarjetas según estilo de juego
                """)
            
        except ImportError:
            st.error("❌ statsbombpy no está instalado")
            st.markdown("""
            **Para instalar statsbombpy:**
            
            ```bash
            pip install statsbombpy
            ```
            
            Luego actualiza tu `requirements.txt`:
            ```
            statsbombpy
            ```
            
            Y redespliega la app en Streamlit Cloud.
            """)
            
            st.info("""
            **¿Qué obtendrás con StatsBomb?**
            - Datos de eventos (pases, tiros, presiones, duelos)
            - Factores de intensidad del juego
            - Estadísticas avanzadas por equipo
            - Ajustes automáticos en predicciones
            """)

else:
    # Mensaje cuando no hay datos suficientes
    st.warning("⚠️ Faltan datos por cargar")
    st.info("Por favor, ve a la pestaña **'📥 Gestión de Datos'** y sincroniza la carpeta local.")
    
    st.markdown("---")
    st.subheader("📋 Archivos Requeridos")
    
    for etiqueta, nombre in archivos_objetivo.items():
        estado = "✅" if etiqueta in st.session_state.get('dfs', {}) else "❌"
        st.write(f"{estado} {etiqueta} (debe empezar con `{nombre}`)")

# ==================== FOOTER ====================

st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])

with col_footer1:
    st.caption("⚽ Sistema de Análisis Estadístico Profesional | Enhanced con StatsBomb")

with col_footer2:
    if st.session_state.get('usar_statsbomb', False):
        st.caption("🌐 StatsBomb: ✅ Activo")
    else:
        st.caption("📊 Modo: Estándar")
