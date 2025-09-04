# Prototipo de Inteligencia Electoral: Manzanillo 🗳️

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0086D1?style=for-the-badge)
![GeoPandas](https://img.shields.io/badge/GeoPandas-139E66?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)

Este repositorio documenta el desarrollo de un prototipo de negocio para el análisis de datos electorales. El objetivo es aplicar técnicas de ingeniería con Modelos de Lenguaje de Gran Escala (LLMs) sobre una base de datos personalizada para facilitar la segmentación estratégica y el micro-targeting de votantes en Manzanillo, Colima.

---

## Proceso de Desarrollo y Metodología

Este prototipo fue construido siguiendo un riguroso proceso de varias fases, desde el tratamiento de datos crudos hasta el despliegue de una interfaz interactiva.

### Fase 1: Consolidación y Auditoría de Datos
El punto de partida fueron tres fuentes de datos dispares: resultados electorales históricos, datos del Censo de Población 2020 a nivel sección, y cartografía geoespacial.
* **Consolidación:** Se unificaron las tres fuentes de datos en un único GeoDataFrame.
* **Validación y Limpieza:** Durante el proceso, se detectaron y resolvieron inconsistencias críticas:
    1.  **Discrepancia de Secciones:** Se resolvió una diferencia entre las 171 secciones del archivo electoral y las 70 del censo, concluyendo que el censo era la fuente de verdad para el municipio.
    2.  **Error Geográfico:** Se diagnosticó y reemplazó un archivo shapefile que correspondía a la Ciudad de México en lugar de Colima.
    3.  **Auditoría de Datos:** Se identificaron y excluyeron secciones con datos anómalos (ej. población total < 50, tasa de participación > 100%) para asegurar la robustez estadística del análisis.

### Fase 2: Ingeniería de Atributos (Feature Engineering)
Para transformar los datos crudos en inteligencia accionable, se diseñó un marco de análisis basado en 4 pilares estratégicos y se crearon nuevas variables:
* **Variables Electorales:** Se calcularon métricas como `pct_voto_oposicion` y un `indice_competitividad` intuitivo (donde 100 es muy competido).
* **Perfiles Sociodemográficos:** Se generó una columna `perfil_descriptivo` para cada sección, asignando etiquetas automáticas (ej. "Predominantemente Jóvenes, Alta Digitalización") basadas en umbrales estadísticos (el 30% superior) de las variables más relevantes del censo.

### Fase 3: Desarrollo del Agente de IA (Ingeniería de LLMs)
El núcleo del prototipo es un agente conversacional que entiende el lenguaje natural.
* **Arquitectura "Text-to-SQL":** Se eligió este enfoque donde el LLM traduce las preguntas del usuario a consultas SQL.
* **Orquestación con LangChain:** Se utilizó la librería LangChain para crear un agente SQL robusto.
* **Ingeniería de Prompts Avanzada:** Se diseñó un prompt detallado que le otorga al LLM una "persona" (`Analista Político Estratégico`) y un **diccionario de datos** que explica el significado semántico de las columnas clave, permitiéndole generar no solo datos, sino insights y recomendaciones.

### Fase 4: Construcción del Prototipo Interactivo
La lógica y el análisis se empaquetaron en una aplicación web fácil de usar.
* **Interfaz con Streamlit:** Se desarrolló la interfaz de usuario completa, incluyendo un layout de dos columnas, widgets interactivos y gestión de estado para la conversación del chat.
* **Visualización Geoespacial:** Se integró un mapa interactivo usando `geopandas.explore` y `streamlit-folium`, con etiquetas permanentes para las secciones y una barra lateral que se actualiza dinámicamente al hacer clic en el mapa.

---

## Características del Prototipo Final

* **Dashboard Geoespacial Interactivo:** Visualiza las 63 secciones electorales auditadas de Manzanillo. Colorea el mapa dinámicamente según variables clave.
* **Perfilamiento Automático:** Cada sección muestra su perfil descriptivo en el mapa.
* **Detalle al Instante:** Al hacer clic en una sección, la barra lateral se actualiza con métricas detalladas de esa zona.
* **Analista Virtual Estratégico:** Un chatbot impulsado por GPT-4o que responde preguntas complejas sobre los datos y ofrece recomendaciones.
* **Filtros Estratégicos:** Permite filtrar y aislar en el mapa las secciones que cumplen con un perfil sociodemográfico específico.

## Configuración e Instalación

Sigue estos pasos para ejecutar el prototipo en tu máquina local.

### 1. Prerrequisitos
* Tener `conda` (se recomienda [Miniforge](https://github.com/conda-forge/miniforge/releases/latest)) instalado.
* Una API Key de OpenAI.

### 2. Clonar el Repositorio
```bash
git clone [https://github.com/](https://github.com/)[TU-USUARIO-DE-GITHUB]/prototipo-inteligencia-electoral.git
cd prototipo-inteligencia-electoral
3. Fuentes de Datos