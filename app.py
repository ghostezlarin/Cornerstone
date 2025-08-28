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
PLANTUML_SERVER_URL = os.environ.get('PLANTUML_SERVER_URL', 'http://plantuml-server:8080')
UML_STORAGE_DIR = 'uml_files'
os.makedirs(UML_STORAGE_DIR, exist_ok=True)


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
    def check_plantuml_jar() -> bool:
        """Check if plantuml.jar exists or server is available"""
        jar_path = PLANTUML_JAR_PATH

        logging.info(f"Checking PlantUML jar at: {jar_path}")
        logging.info(f"PLANTUML_JAR_PATH env variable: {os.environ.get('PLANTUML_JAR_PATH', 'Not set')}")

        # Проверка локального jar
        if os.path.exists(jar_path):
            # Проверка Java
            try:
                result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logging.info(
                        f"Local PlantUML: OK (Java: {result.stderr.splitlines()[0] if result.stderr else 'Unknown'})")
                    return True
            except Exception as e:
                logging.warning(f"Local Java not available: {e}, will use PlantUML server")
        else:
            logging.warning(f"PlantUML jar not found at: {jar_path}")

        # Проверка доступности PlantUML сервера
        try:
            response = requests.get(f"{PLANTUML_SERVER_URL}/", timeout=5)
            if response.status_code == 200:
                logging.info("PlantUML server: OK")
                return True
            else:
                logging.warning(f"PlantUML server responded with status: {response.status_code}")
        except Exception as e:
            logging.warning(f"PlantUML server not available: {e}")

        logging.error("No PlantUML generation method available")
        return False

    @staticmethod
    def generate_svg_with_jar(puml: str, filename_prefix: str) -> Optional[str]:
        """Generates SVG from PlantUML using local plantuml.jar"""
        temp_puml_path = None
        temp_svg_path = None

        try:
            if not puml or len(puml.strip()) < 10:
                logging.error("Empty or too short PlantUML code provided")
                return None

            # Check if plantuml.jar and Java are available
            jar_path = PLANTUML_JAR_PATH
            if not os.path.exists(jar_path):
                logging.error(f"PlantUML jar not found at: {jar_path}")
                return None

            # Проверка Java
            try:
                subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                logging.error(f"Java is not available: {e}")
                return None

            # Sanitize PlantUML code first
            puml = DiagramService.sanitize_puml(puml)

            # Save PlantUML for debugging
            debug_file = DiagramService.save_puml_file(puml, filename_prefix)
            logging.info(f"Saved debug PlantUML to: {debug_file}")

            # Create temporary file with .puml extension
            with tempfile.NamedTemporaryFile(mode='w', suffix='.puml', delete=False, encoding='utf-8') as temp_puml:
                temp_puml.write(puml)
                temp_puml_path = temp_puml.name

            # PlantUML автоматически добавляет .svg к имени входного файла
            base_path = temp_puml_path.rsplit('.', 1)[0]  # Убираем .puml
            temp_svg_path = base_path + '.svg'

            # Run PlantUML to generate SVG
            cmd = [
                'java', '-jar', jar_path,
                '-tsvg',
                '-charset', 'UTF-8',
                '-failfast2',  # Не генерировать SVG при ошибках
                temp_puml_path
            ]

            logging.info(f"Executing PlantUML command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            # Detailed logging
            logging.info(f"PlantUML return code: {result.returncode}")
            if result.stdout and result.stdout.strip():
                logging.info(f"PlantUML stdout: {result.stdout}")
            if result.stderr and result.stderr.strip():
                logging.error(f"PlantUML stderr: {result.stderr}")

            # Проверяем наличие ошибок
            if result.stderr and ("error" in result.stderr.lower() or "fail" in result.stderr.lower()):
                logging.error(f"PlantUML generation failed with errors: {result.stderr}")
                return None

            # Check if SVG file was created
            if os.path.exists(temp_svg_path):
                logging.info(f"SVG file created successfully: {temp_svg_path}")
                with open(temp_svg_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()

                # Validate SVG content
                if '<svg' in svg_content.lower():
                    logging.info("Valid SVG content generated")
                    return svg_content
                else:
                    logging.error("SVG file exists but doesn't contain valid SVG content")
                    logging.error(f"File content starts with: {svg_content[:200]}")
                    return None
            else:
                logging.error(f"SVG file was not created: {temp_svg_path}")
                return None

        except subprocess.TimeoutExpired:
            logging.error("PlantUML generation timed out")
            return None
        except Exception as e:
            logging.error(f"SVG generation with plantuml.jar failed: {str(e)}", exc_info=True)
            return None
        finally:
            # Clean up temporary files
            try:
                if temp_puml_path and os.path.exists(temp_puml_path):
                    os.unlink(temp_puml_path)
                if temp_svg_path and os.path.exists(temp_svg_path):
                    os.unlink(temp_svg_path)
            except OSError as e:
                logging.warning(f"Failed to clean up temp files: {e}")

    @staticmethod
    def generate_svg_with_server(puml: str, filename_prefix: str) -> Optional[str]:
        """Generates SVG using PlantUML server with POST request"""
        try:
            # Проверяем длину PlantUML кода
            if len(puml) > 10000:
                logging.warning(f"PlantUML code too long for server ({len(puml)} chars), using local generation")
                return None

            logging.info("Using PlantUML server for generation with POST")

            # Sanitize PlantUML code first
            puml = DiagramService.sanitize_puml(puml)

            # Save PlantUML for debugging
            debug_file = DiagramService.save_puml_file(puml, filename_prefix)
            logging.info(f"Saved debug PlantUML to: {debug_file}")

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
        """Main SVG generation method - tries local jar first, then server"""
        # Basic validation
        if not puml or len(puml.strip()) < 10:
            logging.error("Invalid or empty PlantUML content")
            return None

        # Всегда сначала пробуем локальную генерацию
        jar_path = PLANTUML_JAR_PATH
        if os.path.exists(jar_path):
            try:
                # Проверяем доступность Java
                subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5, check=True)
                svg = DiagramService.generate_svg_with_jar(puml, filename_prefix)
                if svg:
                    return svg
                logging.warning("Local generation failed")
            except Exception as e:
                logging.warning(f"Java not available for local generation: {e}")

        # Если локальная генерация не удалась, пробуем сервер (только для коротких кодов)
        if len(puml) <= 10000:
            return DiagramService.generate_svg_with_server(puml, filename_prefix)
        else:
            logging.error("PlantUML code too long for server fallback")
            return None


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
        logging.info(f"Retrieved PlantUML content (first 200 chars): {puml[:200]}...")

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
        plantuml_ok = DiagramService.check_plantuml_jar()

        health_status = {
            'status': 'healthy',
            'database': 'connected',
            'plantuml': 'available' if plantuml_ok else 'unavailable',
            'plantuml_jar_path': PLANTUML_JAR_PATH,
            'plantuml_jar_exists': os.path.exists(PLANTUML_JAR_PATH),
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
    logging.info(f"PlantUML jar path: {PLANTUML_JAR_PATH}")
    logging.info(f"PLANTUML_JAR_PATH env: {os.environ.get('PLANTUML_JAR_PATH', 'Not set')}")
    logging.info(f"PlantUML server URL: {PLANTUML_SERVER_URL}")

    # Check database
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            logging.info("Database connection: OK")
    except Exception as e:
        logging.warning(f"Database connection: FAILED - {e}")

    # Check PlantUML
    if DiagramService.check_plantuml_jar():
        logging.info("PlantUML: OK")
    else:
        logging.warning("PlantUML: FAILED - make sure plantuml.jar is available or server is running")

    app.run(host='0.0.0.0', port=5000, debug=True)
