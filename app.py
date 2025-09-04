# app.py - Versión Final y Completa (con índice_competitividad)

import streamlit as st
import geopandas as gpd
import os
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from streamlit_folium import st_folium
from pathlib import Path
import folium

# --- 1. Configuración de la Página ---
st.set_page_config(
    page_title="Análisis Electoral Manzanillo",
    page_icon="🗳️",
    layout="wide"
)

# --- 2. Funciones de Carga y Lógica (Cacheadas para Rendimiento) ---

@st.cache_data
def generar_perfil_seccion(fila, umbrales):
    """Genera una descripción textual del perfil de una sección electoral usando umbrales pre-calculados."""
    perfiles = []
    if fila['porc_jovenes'] > umbrales['Jóvenes']: perfiles.append("Jóvenes")
    if fila['porc_poblacion_migrante'] > umbrales['Migrantes']: perfiles.append("Migrantes")
    if fila['GRAPROES'] > umbrales['Alta Escolaridad']: perfiles.append("Alta Escolaridad")
    if fila['porc_adultos_mayores'] > umbrales['Adultos Mayores']: perfiles.append("Adultos Mayores")
    if fila['indice_digitalizacion'] > umbrales['Alta Digitalización']: perfiles.append("Alta Digitalización")
    return "Predominantemente " + ", ".join(perfiles) if perfiles else "Perfil Mixto / Promedio"

@st.cache_data
def cargar_y_perfilar_datos(ruta_archivo):
    """Carga los datos y aplica el perfilamiento de forma optimizada."""
    try:
        gdf = gpd.read_file(ruta_archivo).to_crs("EPSG:4326")

        # --- NUEVO: Crear índice de competitividad ---
        gdf['indice_competitividad'] = 100 - gdf['competitividad']

        umbrales = {
            'Jóvenes': gdf['porc_jovenes'].quantile(0.70),
            'Migrantes': gdf['porc_poblacion_migrante'].quantile(0.70),
            'Alta Escolaridad': gdf['GRAPROES'].quantile(0.70),
            'Adultos Mayores': gdf['porc_adultos_mayores'].quantile(0.70),
            'Alta Digitalización': gdf['indice_digitalizacion'].quantile(0.70),
        }
        gdf['perfil_descriptivo'] = gdf.apply(generar_perfil_seccion, axis=1, umbrales=umbrales)
        return gdf
    except Exception as e:
        st.error(f"Error al cargar y perfilar los datos: {e}")
        return None

@st.cache_resource
def inicializar_agente(_df):
    """Inicializa la DB y el agente SQL con el prompt estratégico."""
    try:
        df_analisis = _df.drop(columns=['geometry'], errors='ignore')
        engine = create_engine('sqlite:///manzanillo_data.db')
        df_analisis.to_sql('secciones', engine, index=False, if_exists='replace')
        db = SQLDatabase(engine=engine)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # --- INICIO DEL PROMPT COMPLETO Y RESTAURADO ---
        prompt_personalizado = """
### Persona y Tarea Principal
Eres "Analista Político Estratégico", un asistente de IA experto en el análisis de datos electorales y sociodemográficos de Manzanillo, Colima.
Tu única tarea es responder a las preguntas del usuario generando y ejecutando consultas SQL sobre una tabla llamada 'secciones', y luego interpretar los resultados de forma clara y analítica.

### Diccionario de Datos Clave
Aquí tienes el significado de las columnas más importantes para tu análisis:
- seccion: El identificador único de la sección electoral.
- partido_dominante: El partido político con más votos históricos en la sección.
- pct_voto_morena, pct_voto_oposicion: Porcentaje de votos para Morena y la oposición.
- tasa_participacion_promedio: Porcentaje de ciudadanos registrados que votan. Es un indicador clave de compromiso cívico.
- indice_competitividad: Un puntaje de 0 a 100 que mide qué tan reñida es una elección. IMPORTANTE: un valor ALTO (cercano a 100) significa MUY COMPETITIVO. Un valor BAJO significa que un partido domina.
- porc_jovenes: Porcentaje de la población entre 18 y 24 años.
- porc_adultos_mayores: Porcentaje de la población mayor a 65 años.
- indice_digitalizacion: Un puntaje de 0 a 100 que mide la adopción tecnológica (internet, PC, celular). Es un indicador de modernidad.
- GRAPROES: Grado promedio de escolaridad en años. Un indicador socioeconómico clave.
- porc_hogares_jefa_mujer: Porcentaje de hogares liderados por una mujer.
- porc_poblacion_migrante: Porcentaje de residentes nacidos fuera de Colima. Indica arraigo comunitario o dinamismo poblacional.
- tasa_desocupacion: Porcentaje de la población económicamente activa que está desempleada.
- porc_sin_servicios_salud: Porcentaje de la población sin acceso a servicios de salud. Un indicador clave de vulnerabilidad.

### Instrucciones de Salida
1. Analiza la pregunta del usuario para entender su intención estratégica.
2. Usa el diccionario de datos para elegir las mejores columnas para tu consulta SQL.
3. Genera una consulta SQL en dialecto SQLite.
4. Una vez que tengas los resultados, no te limite a mostrarlos. Escribe un resumen ejecutivo en español, explicando los hallazgos y dando recomendaciones prácticas de estrategia electoral o política pública.
5. Si presentas una lista de secciones, usa un formato de viñetas (bullets).
6. Siempre responde en español y actúa como un analista político experimentado.
"""
        # --- FIN DEL PROMPT ---
        
        agente = create_sql_agent(llm=llm, db=db, agent_type="openai-tools", verbose=False, prompt_suffix=prompt_personalizado)
        return agente
    except Exception as e:
        st.error(f"Error al inicializar el agente LLM: {e}")
        return None

# --- 3. APLICACIÓN PRINCIPAL ---

st.title("Prototipo de Inteligencia Electoral: Manzanillo")
st.markdown("Analiza datos seccionales con mapas interactivos, KPIs y consultas inteligentes con IA.")


DIRECTORIO_SCRIPT = Path(__file__).parent
RUTA_DATOS_FINAL = DIRECTORIO_SCRIPT / "1_datos" / "02_procesados" / "gdf_final_auditado.gpkg"
gdf_data = cargar_y_perfilar_datos(RUTA_DATOS_FINAL)

if gdf_data is not None:
    
    with st.sidebar:
        st.header("🎛️ Controles del Mapa")
        perfiles_unicos = sorted(gdf_data['perfil_descriptivo'].unique())
        opciones_filtro = ["— Mostrar Todas las Secciones —"] + perfiles_unicos
        perfil_seleccionado = st.selectbox("Filtrar por Perfil Sociodemográfico:", options=opciones_filtro)
        st.caption("Selecciona un perfil para aislarlo en el mapa.")
        st.divider()
        opciones_visualizacion = {
            'Tasa de Participación (%)': 'tasa_participacion_promedio',
            'Porcentaje Voto Morena': 'pct_voto_morena',
            'Índice de Competitividad': 'indice_competitividad',  # <--- Nuevo índice
            'Índice de Digitalización': 'indice_digitalizacion',
        }
        opcion_seleccionada_nombre = st.selectbox("Variable a visualizar:", options=list(opciones_visualizacion.keys()))
        columna_a_visualizar = opciones_visualizacion[opcion_seleccionada_nombre]
        st.divider()
        st.header("📌 Detalle de Sección")
        detalle_placeholder = st.empty()

    if perfil_seleccionado == "— Mostrar Todas las Secciones —":
        gdf_filtrado = gdf_data
    else:
        gdf_filtrado = gdf_data[gdf_data['perfil_descriptivo'] == perfil_seleccionado]

    col_mapa, col_chat = st.columns([2, 1])

    with col_mapa:
        st.subheader("🗺️ Exploración Geoespacial")
        st.info(f"Mostrando **{len(gdf_filtrado)}** de **{len(gdf_data)}** secciones.")
        m = gdf_filtrado.explore(
            column=columna_a_visualizar, cmap='plasma', tooltip=['seccion', 'perfil_descriptivo'],
            popup=False, legend=True, scheme='quantiles',
            legend_kwds={'caption': opcion_seleccionada_nombre},
            style_kwds={'stroke': True, 'color': 'black', 'weight': 0.6},
            tiles="CartoDB positron"
        )
        for idx, row in gdf_filtrado.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(
                    icon_size=(150,36), icon_anchor=(7,20),
                    html=f'<div style="font-size: 11pt; font-weight: bold; color: #333; text-shadow: 1px 1px 2px white;">{row.seccion}</div>',
                )
            ).add_to(m)
        map_data = st_folium(m, use_container_width=True, height=600)

    # --- Chat con agente ---
    with col_chat:
        col_titulo, col_limpiar = st.columns([3, 1])
        with col_titulo:
            st.subheader("🤖 Analista Virtual Estratégico")
        with col_limpiar:
            if st.button("🗑️ Limpiar Chat", help="Borrar historial de conversación"):
                st.session_state.messages = [{"role": "assistant", "content": "Hola, soy tu analista estratégico. ¿Qué necesitas evaluar?"}]
                st.rerun()
        
        agente_sql = inicializar_agente(gdf_data)
        if agente_sql:
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Hola, soy tu analista estratégico. ¿Qué necesitas evaluar?"}]
            
            if len(st.session_state.messages) > 10:
                st.warning("💡 El historial está largo. Considera limpiar el chat para mejor rendimiento.")
            
            chat_container = st.container(height=400)
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            if prompt := st.chat_input("Ej: Secciones más competitivas..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Analizando y formulando estrategia..."):
                            response = agente_sql.invoke(prompt)
                            respuesta_texto = response['output']
                            st.markdown(respuesta_texto)
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
                st.rerun()

    # --- Panel de detalle ---
    seccion_seleccionada_data = None
    if map_data and map_data.get("last_active_drawing"):
        properties = map_data["last_active_drawing"]["properties"]
        seccion_seleccionada_data = gdf_data[gdf_data['seccion'] == properties['seccion']].iloc[0]
    
    with detalle_placeholder.container():
        if seccion_seleccionada_data is not None:
            seccion_id = seccion_seleccionada_data.get('seccion', 'N/A')
            perfil = seccion_seleccionada_data.get('perfil_descriptivo', 'No disponible')
            partido_dom = seccion_seleccionada_data.get('partido_dominante', 'N/A')
            participacion = seccion_seleccionada_data.get('tasa_participacion_promedio', 0.0)
            voto_morena = seccion_seleccionada_data.get('pct_voto_morena', 0.0)
            voto_oposicion = seccion_seleccionada_data.get('pct_voto_oposicion', 0.0)
            indice_competitividad = seccion_seleccionada_data.get('indice_competitividad', 0.0)
            escolaridad = seccion_seleccionada_data.get('GRAPROES', 0.0)
            digitalizacion = seccion_seleccionada_data.get('indice_digitalizacion', 0.0)
            jovenes = seccion_seleccionada_data.get('porc_jovenes', 0.0)
            adultos_mayores = seccion_seleccionada_data.get('porc_adultos_mayores', 0.0)
            migrantes = seccion_seleccionada_data.get('porc_poblacion_migrante', 0.0)
            hogares_jefa_mujer = seccion_seleccionada_data.get('porc_hogares_jefa_mujer', 0.0)
            sin_servicios_salud = seccion_seleccionada_data.get('porc_sin_servicios_salud', 0.0)
            desocupacion = seccion_seleccionada_data.get('tasa_desocupacion', 0.0)
            
            st.subheader(f"📍 Sección {seccion_id}")
            st.info(f"**Perfil:** {perfil}")
            
            if isinstance(partido_dom, str) and partido_dom != 'N/A':
                st.metric(label="Partido Dominante", value=partido_dom.title())
            
            st.markdown("### 🗳️ **Indicadores Electorales**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Participación Electoral",
                    value=f"{participacion:.1f}%",
                    help="Porcentaje de ciudadanos registrados que votan"
                )
                
            with col2:
                st.metric(
                    label="Voto Morena",
                    value=f"{voto_morena:.1f}%",
                    help="Porcentaje histórico de votos para Morena"
                )
                
            with col3:
                if indice_competitividad >= 80:
                    comp_texto = "🔥 MUY Alta"
                    comp_desc = "Campo de batalla electoral"
                elif indice_competitividad >= 60:
                    comp_texto = "⚡ Alta"
                    comp_desc = "Zona de disputa"
                elif indice_competitividad >= 40:
                    comp_texto = "🟡 Media"
                    comp_desc = "Moderadamente disputada"
                else:
                    comp_texto = "🏛️ Baja"
                    comp_desc = "Sección consolidada"
                    
                st.metric(
                    label="Índice de Competitividad",
                    value=comp_texto,
                    help=f"{comp_desc}. Valor calculado: {indice_competitividad:.1f} (0 = no competitivo, 100 = muy competitivo)"
                )
            
            # --- Perfil sociodemográfico ---
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📚 Educación y Tecnología**
                - Escolaridad promedio: **{escolaridad:.1f} años**
                - Índice digitalización: **{digitalizacion:.0f}/100**
                
                **🏠 Composición Poblacional**
                - Jóvenes (18-24): **{jovenes:.1f}%**
                - Adultos mayores (+65): **{adultos_mayores:.1f}%**
                - Población migrante: **{migrantes:.1f}%**
                """)
            with col2:
                st.markdown(f"""
                **💼 Indicadores Socioeconómicos**
                - Hogares con jefa mujer: **{hogares_jefa_mujer:.1f}%**
                - Tasa de desocupación: **{desocupacion:.1f}%**
                - Sin servicios de salud: **{sin_servicios_salud:.1f}%**
                
                **⚖️ Balance Político**
                - Voto oposición: **{voto_oposicion:.1f}%**
                """)
            
            # --- Insights rápidos ---
            st.markdown("### 🎯 **Análisis Estratégico Rápido**")
            insights = []
            if participacion > 70:
                insights.append("🟢 **Alta participación cívica** - Ciudadanía comprometida")
            elif participacion < 50:
                insights.append("🔴 **Baja participación** - Oportunidad de movilización")
            else:
                insights.append("🟡 **Participación moderada** - Potencial de crecimiento")
                
            if indice_competitividad >= 80:
                insights.append("🔥 **Sección muy competitiva** - Campo de batalla, cada voto cuenta")
            elif indice_competitividad >= 60:
                insights.append("⚡ **Sección competitiva** - Zona de disputa electoral")
            elif indice_competitividad <= 30:
                insights.append("🏛️ **Dominio partidista** - Sección consolidada")
                
            if digitalizacion > 70:
                insights.append("📱 **Alta conectividad** - Estrategias digitales efectivas")
            elif digitalizacion < 30:
                insights.append("📻 **Baja digitalización** - Enfocar en medios tradicionales")
                
            if jovenes > 25:
                insights.append("👨‍🎓 **Población joven** - Mensajes de cambio y oportunidad")
            if adultos_mayores > 15:
                insights.append("👴 **Población envejecida** - Mensajes de estabilidad y seguridad")
            if sin_servicios_salud > 20:
                insights.append("🏥 **Vulnerabilidad en salud** - Priorizar políticas sanitarias")
            
            # CORRECCIÓN: Esta línea estaba mal indentada
            for insight in insights:
                st.markdown(f"- {insight}")
                
            # Botón para análisis detallado
            if st.button(f"🔍 Análisis Detallado Sección {seccion_id}", use_container_width=True):
                # Trigger análisis con el chatbot
                consulta_detallada = (
                    f"Analiza en detalle la sección {seccion_id}, incluyendo fortalezas, "
                    f"debilidades y recomendaciones estratégicas específicas"
                )
                st.session_state.messages.append({"role": "user", "content": consulta_detallada})
                
        else:
            st.info("Haz clic en una sección del mapa para ver sus detalles aquí.")
            st.markdown("""
            ### 💡 **Cómo usar este panel:**
            1. **Haz clic** en cualquier sección del mapa
            2. **Visualiza** métricas electorales y sociodemográficas
            3. **Obtén insights** estratégicos automáticos
            4. **Solicita análisis** detallado con el botón
            
            ### 📊 **Variables disponibles:**
            - Indicadores electorales (participación, preferencias, **índice de competitividad**)
            - Demografía (edad, migración, género)
            - Socioeconómicos (educación, empleo, salud)
            - Tecnológicos (digitalización)
            """)
else:
    st.warning("⚠️ No se pudieron cargar los datos.")