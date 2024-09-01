# PDF-Query-Engine

## Descripción

PDF-Query-Engine es una aplicación basada en modelos de lenguaje (LLM) diseñada para interactuar con archivos PDF. Permite cargar documentos PDF, extraer su contenido y responder a preguntas sobre el texto de esos documentos. Utiliza técnicas avanzadas de procesamiento de lenguaje natural para ofrecer respuestas precisas y concisas a preguntas específicas.

## Funcionalidades

- Carga y procesamiento de archivos PDF.
- Extracción y segmentación del texto contenido en los PDF.
- Respuesta a preguntas sobre el contenido del PDF utilizando un modelo de lenguaje.
- Configuración de claves API a través de variables de entorno.
- Registro detallado y manejo de errores durante la ejecución.

## Instalación

1. **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/PDF-Query-Engine.git
    ```

2. **Navega al directorio del proyecto:**
    ```bash
    cd PDF-Query-Engine
    ```

3. **Crea un entorno virtual e instala las dependencias:**
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    pip install -r requirements.txt
    ```

4. **Configura las variables de entorno:**
    Crea un archivo `.env` en el directorio raíz del proyecto y añade la clave API de OpenAI:
    ```env
    API_KEY_OPENAI=tu_clave_api
    ```

## Uso

1. **Asegúrate de tener un archivo PDF en el directorio `data`.**

2. **Ejecuta el script principal:**
    ```bash
    python src/main.py
    ```

3. **Para hacer preguntas sobre el contenido del PDF, sigue las instrucciones en el script para definir las preguntas a realizar.**

## Estructura del Proyecto

- `src/`
  - `main.py` - Script principal para cargar el PDF y realizar preguntas.
  - `config.py` - Configuración del proyecto, incluyendo carga de variables de entorno y configuración de logging.
- `data/` - Directorio para almacenar archivos PDF.
- `requirements.txt` - Dependencias del proyecto.
- `.env` - Archivo para configurar variables de entorno.

## Contribuciones

Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tus cambios.
3. Realiza los cambios y haz commits.
4. Envía un pull request con una descripción detallada de los cambios.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto

Para más información o preguntas, puedes contactarme a través de [tu_email@ejemplo.com].

