import logging
import re
import os
import json
import subprocess
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime
from flask import Flask, Response, request, redirect
import psycopg2
import requests

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagram_service.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Database configuration - используем переменные окружения
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '*'),
    'port': os.environ.get('DB_PORT', '*'),
    'dbname': os.environ.get('DB_NAME', '*'),
    'user': os.environ.get('DB_USER', '*'),
    'password': os.environ.get('DB_PASSWORD', '*')
}

# PlantUML configuration
PLANTUML_JAR_PATH = os.environ.get('PLANTUML_JAR_PATH', 'plantuml.jar')
PLANTUML_SERVER_URL = os.environ.get('PLANTUML_SERVER_URL', 'http://localhost:8080')
UML_STORAGE_DIR = 'uml_files'
os.makedirs(UML_STORAGE_DIR, exist_ok=True)

# Глобальная переменная для кэширования PlantUML кода
PLANTUML_CACHE = {}


class DiagramService:
    @staticmethod
    def normalize_guid(guid_str: str) -> Optional[str]:
        """Normalizes GUID to g_xxxxxxxx format"""
        try:
            if not guid_str:
                return None

            clean_guid = re.sub(r'[^a-fA-F0-9]', '', guid_str.lower())
            guid_part = clean_guid[1:] if clean_guid.startswith('g') else clean_guid

            if len(guid_part) == 32:
                formatted = f"{guid_part[:8]}-{guid_part[8:12]}-{guid_part[12:16]}-{guid_part[16:20]}-{guid_part[20:32]}"
                if re.fullmatch(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', formatted):
                    normalized = f"g_{formatted}"
                    logging.info(f"Normalized GUID: {guid_str} -> {normalized}")
                    return normalized
            logging.warning(f"Invalid GUID format: {guid_str}")
            return None
        except Exception as e:
            logging.error(f"GUID normalization failed: {str(e)}")
            return None

    @staticmethod
    def get_puml_from_db(params: Dict[str, Any]) -> Optional[str]:
        """Gets PlantUML code from database using f_GetDiagram2 function with JSONB parameters"""
        try:
            # Создаем ключ для кэширования
            cache_key = json.dumps(params, sort_keys=True)

            # Проверяем кэш
            if cache_key in PLANTUML_CACHE:
                logging.info(f"Cache hit for key: {cache_key}")
                return PLANTUML_CACHE[cache_key]

            logging.info(f"Cache miss for key: {cache_key}")

            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    # Prepare parameters for JSONB function - используем все переданные параметры
                    db_params = params.copy()

                    # Normalize GUID if present
                    if 'guid' in db_params and db_params['guid']:
                        guid = db_params['guid']
                        normalized_guid = DiagramService.normalize_guid(guid)
                        if normalized_guid:
                            db_params['guid'] = normalized_guid[2:]  # Remove 'g_' prefix for DB
                        else:
                            logging.warning(f"Invalid GUID: {guid}, using default")
                            db_params['guid'] = '00000000-0000-0000-0000-000000000001'

                    logging.info(f"Calling f_GetDiagram2 with params: {db_params}")

                    cur.execute("""
                        SELECT main."f_GetDiagram2"(%s::jsonb)
                    """, (json.dumps(db_params),))

                    result = cur.fetchone()
                    if result and result[0]:
                        # Сохраняем в кэш
                        PLANTUML_CACHE[cache_key] = result[0]
                        return result[0]
                    else:
                        logging.warning("No result returned from database")
                        return None
            return None
        except psycopg2.Error as e:
            logging.error(f"Database error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in database query: {str(e)}")
            return None

    @staticmethod
    def save_puml_file(puml: str, filename_prefix: str) -> str:
        """Saves PlantUML code to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', filename_prefix)
            filename = f"{UML_STORAGE_DIR}/{safe_prefix}_{timestamp}.puml"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(puml)
            logging.info(f"Saved PlantUML to: {filename}")
            return filename
        except Exception as e:
            logging.error(f"Failed to save file: {str(e)}")
            return ""

    @staticmethod
    def sanitize_puml(puml: str) -> str:
        """Cleans PlantUML code before processing"""
        if not puml:
            return puml

        fixes = [
            (r'\\"', '"'),
            (r'\\{\\}', ''),
            (r'\\./', './'),
            (r'include <c4/', '!include <C4/'),  # Исправляем регистр
            (r'#include', '!include'),  # Исправляем неправильные директивы
            (r'linclude', '!include'),  # Исправляем опечатки
            (r'Lunquoted', '!unquoted'),  # Исправляем опечатки
            (r'lendprocedure', '!endprocedure'),  # Исправляем опечатки
            (r'!!include', '!include'),  # Исправляем двойной !
        ]

        for pattern, replacement in fixes:
            puml = re.sub(pattern, replacement, puml)

        return puml

    @staticmethod
    def check_plantuml_server() -> bool:
        """Check if PlantUML server is available"""
        try:
            response = requests.get(f"{PLANTUML_SERVER_URL}/", timeout=5)
            if response.status_code == 200:
                logging.info("PlantUML server: OK")
                return True
            else:
                logging.warning(f"PlantUML server responded with status: {response.status_code}")
                return False
        except Exception as e:
            logging.warning(f"PlantUML server not available: {e}")
            return False

    @staticmethod
    def generate_svg_with_server(puml: str, filename_prefix: str) -> Optional[str]:
        """Generates SVG using PlantUML server with POST request"""
        try:
            # Проверяем длину PlantUML кода
            if len(puml) > 100000:  # Увеличиваем лимит для сервера
                logging.warning(f"PlantUML code too long for server ({len(puml)} chars)")
                return None

            logging.info("Using PlantUML server for generation with POST")

            # Sanitize PlantUML code first
            puml = DiagramService.sanitize_puml(puml)

            # Save PlantUML for debugging (только для отладки)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                debug_file = DiagramService.save_puml_file(puml, filename_prefix)
                logging.debug(f"Saved debug PlantUML to: {debug_file}")

            # Формируем URL для PlantUML сервера (POST endpoint)
            plantuml_url = f"{PLANTUML_SERVER_URL}/svg"

            logging.info(f"Requesting PlantUML server via POST: {plantuml_url}")
            logging.info(f"PlantUML content length: {len(puml)} characters")

            # Используем POST запрос вместо GET чтобы избежать длинных URL
            headers = {'Content-Type': 'text/plain; charset=utf-8'}
            response = requests.post(plantuml_url, data=puml.encode('utf-8'),
                                     headers=headers, timeout=30)

            # Проверяем, если сервер возвращает SVG даже при статусе 400
            if response.status_code in [200, 400]:  # Принимаем и 400 статус
                svg_content = response.text
                if '<svg' in svg_content.lower():
                    logging.info(f"PlantUML server generation successful (status {response.status_code})")
                    return svg_content
                else:
                    logging.error("PlantUML server returned invalid content")
                    logging.error(f"Response content starts with: {svg_content[:200]}")
            else:
                logging.error(f"PlantUML server failed - status: {response.status_code}")
                logging.error(f"Response text: {response.text[:200]}")

            return None

        except requests.exceptions.Timeout:
            logging.error("PlantUML server request timed out")
            return None
        except Exception as e:
            logging.error(f"PlantUML server generation failed: {str(e)}")
            return None

    @staticmethod
    def generate_svg(puml: str, filename_prefix: str) -> Optional[str]:
        """Main SVG generation method - uses PlantUML server only"""
        # Basic validation
        if not puml or len(puml.strip()) < 10:
            logging.error("Invalid or empty PlantUML content")
            return None

        # Всегда используем PlantUML сервер
        return DiagramService.generate_svg_with_server(puml, filename_prefix)


@app.route('/')
def index():
    """Redirect root to default diagram with GUID 0001 and type WBS"""
    default_guid = '00000000-0000-0000-0000-000000000001'
    return redirect(f'/api/diagram?guid={default_guid}&type=WBS&level=1')


@app.route('/diagram')
def legacy_diagram():
    """Redirect legacy /diagram endpoint to /api/diagram"""
    # Сохраняем все параметры из запроса
    params = request.args.to_dict()
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])

    if query_string:
        return redirect(f'/api/diagram?{query_string}')
    else:
        # Если параметров нет, используем стандартные
        default_guid = '00000000-0000-0000-0000-000000000001'
        return redirect(f'/api/diagram?guid={default_guid}&type=WBS&level=1')


@app.route('/api/diagram')
def get_diagram_svg():
    """Universal endpoint for diagram generation with JSON parameters"""
    try:
        # Get parameters from JSON body or query string
        if request.is_json:
            params = request.get_json()
        else:
            params = request.args.to_dict()

        # Convert level to int if present
        if 'level' in params:
            try:
                params['level'] = int(params['level'])
            except ValueError:
                params['level'] = 1

        logging.info(f"Processing diagram request with params: {params}")

        # Validate required parameters
        if 'guid' not in params or not params['guid']:
            return Response("Missing required parameter: guid", status=400)

        puml = DiagramService.get_puml_from_db(params)
        if not puml:
            return Response("Diagram not found", status=404)

        # Log the PlantUML content for debugging
        logging.debug(f"Retrieved PlantUML content (first 200 chars): {puml[:200]}...")

        # Create filename prefix for logging
        diagram_type = params.get('type', 'unknown')
        guid = params.get('guid', 'unknown')
        filename_prefix = f"{diagram_type}_{guid}"

        svg = DiagramService.generate_svg(puml, filename_prefix)
        if not svg:
            return Response("SVG generation failed", status=500)

        return Response(svg, mimetype='image/svg+xml')

    except Exception as e:
        logging.exception("Unexpected error in diagram endpoint")
        return Response("Internal server error", status=500)


@app.route('/api/diagram/debug')
def debug_diagram():
    """Debug endpoint to get raw PlantUML code"""
    try:
        # Get parameters from JSON body or query string
        if request.is_json:
            params = request.get_json()
        else:
            params = request.args.to_dict()

        if 'level' in params:
            try:
                params['level'] = int(params['level'])
            except ValueError:
                params['level'] = 1

        logging.info(f"Debug request with params: {params}")

        if 'guid' not in params or not params['guid']:
            return Response("Missing required parameter: guid", status=400)

        puml = DiagramService.get_puml_from_db(params)
        if not puml:
            return Response("Diagram not found", status=404)

        return Response(puml, mimetype='text/plain')

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

        # Check PlantUML availability
        plantuml_ok = DiagramService.check_plantuml_server()

        health_status = {
            'status': 'healthy',
            'database': 'connected',
            'plantuml': 'available' if plantuml_ok else 'unavailable',
            'plantuml_server_url': PLANTUML_SERVER_URL,
            'timestamp': datetime.now().isoformat()
        }

        return Response(json.dumps(health_status, indent=2), mimetype='application/json')

    except Exception as e:
        return Response(json.dumps({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), status=500, mimetype='application/json')


# Backward compatibility endpoints
@app.route('/api/c4')
def get_c4_svg():
    """Backward compatibility for C4 diagrams"""
    # Get parameters from JSON body or query string
    if request.is_json:
        params = request.get_json()
    else:
        params = request.args.to_dict()
    params['type'] = 'C4'
    return get_diagram_svg()


@app.route('/api/wbs')
def get_wbs_svg():
    """Backward compatibility for WBS diagrams"""
    # Get parameters from JSON body or query string
    if request.is_json:
        params = request.get_json()
    else:
        params = request.args.to_dict()
    params['type'] = 'WBS'
    return get_diagram_svg()


if __name__ == '__main__':
    # Check dependencies on startup
    logging.info("Starting Diagram Service...")
    logging.info("Checking dependencies:")
    logging.info(f"PlantUML server URL: {PLANTUML_SERVER_URL}")

    # Check database
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            logging.info("Database connection: OK")
    except Exception as e:
        logging.warning(f"Database connection: FAILED - {e}")

    # Check PlantUML server
    if DiagramService.check_plantuml_server():
        logging.info("PlantUML server: OK")
    else:
        logging.warning("PlantUML server: FAILED - make sure PlantUML server is running")

    app.run(host='0.0.0.0', port=5000, debug=False)  # Отключаем debug для production