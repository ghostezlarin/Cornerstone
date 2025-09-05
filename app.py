import logging
import re
import os
import json
import time
import threading
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from flask import Flask, Response, request, redirect, jsonify
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
PLANTUML_SERVER_URL = os.environ.get('PLANTUML_SERVER_URL', 'http://plantuml-server:8080')
UML_STORAGE_DIR = 'uml_files'
os.makedirs(UML_STORAGE_DIR, exist_ok=True)

# Глобальная переменная для кэширования PlantUML кода с timestamp (TTL = 30 секунд)
PLANTUML_CACHE = {}
CACHE_TTL_SECONDS = 30


class DiagramService:
    @staticmethod
    def normalize_guid(guid_str: str) -> Optional[str]:
        """Normalizes GUID to g_xxxxxxxx format"""
        try:
            if not guid_str:
                return None

            # Убираем все не-hex символы
            clean_guid = re.sub(r'[^a-fA-F0-9]', '', guid_str.lower())

            # Если GUID уже в формате без дефисов
            if len(clean_guid) == 32:
                # Форматируем с дефисами
                formatted = f"{clean_guid[:8]}-{clean_guid[8:12]}-{clean_guid[12:16]}-{clean_guid[16:20]}-{clean_guid[20:32]}"
                if re.fullmatch(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', formatted):
                    normalized = f"g_{formatted}"
                    logging.info(f"Normalized GUID: {guid_str} -> {normalized}")
                    return normalized

            # Если GUID уже в правильном формате с дефисами
            if re.fullmatch(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', guid_str.lower()):
                normalized = f"g_{guid_str.lower()}"
                logging.info(f"Normalized GUID (already formatted): {guid_str} -> {normalized}")
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

            # Проверяем кэш (TTL = 30 секунд)
            if cache_key in PLANTUML_CACHE:
                cached_data = PLANTUML_CACHE[cache_key]
                cache_time, puml_code = cached_data

                # Проверяем не истек ли TTL
                if time.time() - cache_time < CACHE_TTL_SECONDS:
                    logging.info(f"Cache hit for key: {cache_key} (age: {time.time() - cache_time:.1f}s)")
                    return puml_code
                else:
                    logging.info(f"Cache expired for key: {cache_key} (age: {time.time() - cache_time:.1f}s)")
                    del PLANTUML_CACHE[cache_key]  # Удаляем просроченный кэш

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
                        puml_code = result[0]

                        # Проверяем, не вернула ли БД сообщение об ошибке
                        error_indicators = ['error', 'ошибка', 'fail', 'exception', 'invalid']
                        if any(indicator in puml_code.lower() for indicator in error_indicators):
                            logging.error(f"Database returned possible error: {puml_code[:100]}...")
                            # Не кэшируем ошибки!
                            return None

                        # Сохраняем в кэш с timestamp
                        PLANTUML_CACHE[cache_key] = (time.time(), puml_code)
                        logging.info(f"Cached result for key: {cache_key}")
                        return puml_code
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
            return Response("Diagram not found or database error", status=404)

        # Проверяем, не является ли результат ошибкой
        error_indicators = ['error', 'ошибка', 'fail', 'exception', 'invalid']
        if any(indicator in puml.lower() for indicator in error_indicators):
            logging.error(f"Database returned error content: {puml[:200]}...")
            return Response("Database returned error content", status=500)

        # Log the PlantUML content for debugging
        logging.debug(f"Retrieved PlantUML content (first 200 chars): {puml[:200]}...")
        logging.info(f"Retrieved PlantUML content length: {len(puml)} characters")

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


@app.route('/api/diagram/debug/guid')
def debug_guid():
    """Debug endpoint for GUID normalization"""
    guid = request.args.get('guid', '')
    normalized = DiagramService.normalize_guid(guid)
    return jsonify({
        'input': guid,
        'normalized': normalized,
        'is_valid': normalized is not None
    })


@app.route('/api/diagram/debug/db')
def debug_db_query():
    """Debug endpoint to test database query directly"""
    try:
        guid = request.args.get('guid', '00000000-0000-0000-0000-000000000001')
        diagram_type = request.args.get('type', 'WBS')

        params = {
            'guid': guid,
            'type': diagram_type,
            'level': int(request.args.get('level', 1))
        }

        # Test normalization
        normalized_guid = DiagramService.normalize_guid(guid)

        # Test database query
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                db_params = params.copy()
                if normalized_guid:
                    db_params['guid'] = normalized_guid[2:]  # Remove 'g_' prefix

                cur.execute("""
                    SELECT main."f_GetDiagram2"(%s::jsonb)
                """, (json.dumps(db_params),))

                result = cur.fetchone()

        return jsonify({
            'input_params': params,
            'normalized_guid': normalized_guid,
            'db_params': db_params,
            'db_result': result[0] if result else None,
            'result_length': len(result[0]) if result else 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache/status')
def cache_status():
    """Get cache status and statistics"""
    cache_size = len(PLANTUML_CACHE)
    now = time.time()

    # Статистика по возрасту записей
    age_stats = []
    for key, (cache_time, _) in PLANTUML_CACHE.items():
        age = now - cache_time
        age_stats.append(age)

    return jsonify({
        'cache_size': cache_size,
        'max_ttl_seconds': CACHE_TTL_SECONDS,
        'oldest_entry_seconds': max(age_stats) if age_stats else 0,
        'newest_entry_seconds': min(age_stats) if age_stats else 0,
        'average_age_seconds': sum(age_stats) / len(age_stats) if age_stats else 0,
        'expired_entries': sum(1 for age in age_stats if age >= CACHE_TTL_SECONDS) if age_stats else 0
    })


@app.route('/api/cache/clear')
def clear_cache():
    """Clear all cached data"""
    global PLANTUML_CACHE
    cleared_count = len(PLANTUML_CACHE)
    PLANTUML_CACHE = {}
    logging.info(f"Cache cleared, removed {cleared_count} entries")
    return jsonify({
        'cleared_entries': cleared_count,
        'status': 'cache_cleared'
    })


@app.route('/api/cache/cleanup')
def cleanup_cache():
    """Remove expired cache entries"""
    global PLANTUML_CACHE
    now = time.time()
    expired_count = 0

    keys_to_remove = []
    for key, (cache_time, _) in PLANTUML_CACHE.items():
        if now - cache_time >= CACHE_TTL_SECONDS:
            keys_to_remove.append(key)
            expired_count += 1

    for key in keys_to_remove:
        del PLANTUML_CACHE[key]

    logging.info(f"Cache cleanup removed {expired_count} expired entries")
    return jsonify({
        'removed_entries': expired_count,
        'remaining_entries': len(PLANTUML_CACHE),
        'status': 'cleanup_completed'
    })


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
            'cache_size': len(PLANTUML_CACHE),
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


def periodic_cache_cleanup():
    """Background thread to periodically clean expired cache entries"""
    while True:
        time.sleep(60)  # Check every minute
        try:
            now = time.time()
            expired_count = 0

            keys_to_remove = []
            for key, (cache_time, _) in PLANTUML_CACHE.items():
                if now - cache_time >= CACHE_TTL_SECONDS:
                    keys_to_remove.append(key)
                    expired_count += 1

            for key in keys_to_remove:
                del PLANTUML_CACHE[key]

            if expired_count > 0:
                logging.info(f"Background cache cleanup removed {expired_count} expired entries")

        except Exception as e:
            logging.error(f"Error in background cache cleanup: {e}")


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

    # Запускаем фоновый поток для очистки кэша
    cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
    cleanup_thread.start()
    logging.info("Background cache cleanup thread started")

    app.run(host='0.0.0.0', port=5000, debug=False)