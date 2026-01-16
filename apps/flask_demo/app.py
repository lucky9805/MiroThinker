
import asyncio
import json
import logging
import os
import threading
import time
import sys
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from flask import Flask, request, jsonify, Response, render_template, stream_with_context

# Add parent directory to sys.path to allow imports from utils and sibling modules if needed
current_dir = Path(__file__).parent.resolve()
apps_dir = current_dir.parent.resolve()
gradio_demo_dir = apps_dir / "gradio-demo"
miroflow_agent_dir = apps_dir / "miroflow-agent"

sys.path.append(str(current_dir))
if gradio_demo_dir.exists():
    sys.path.append(str(gradio_demo_dir))
if miroflow_agent_dir.exists():
    sys.path.append(str(miroflow_agent_dir))

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Import from parent directory modules (assuming they are accessible via sys.path)
try:
    from prompt_patch import apply_prompt_patch
    from src.config.settings import expose_sub_agents_as_tools
    from src.core.pipeline import create_pipeline_components, execute_task_pipeline
    from utils import replace_chinese_punctuation
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Import Error: {e}")
    logger.info(f"sys.path: {sys.path}")
    # Try one more time adding root if needed (though miroflow-agent should cover src)
    if "src" in str(e):
         # Maybe src is in root?
         pass
    raise e

from flask_cors import CORS

# Apply custom system prompt patch (adds MiroThinker identity)
apply_prompt_patch()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global cleanup thread pool
cleanup_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cleanup")

# Set DEMO_MODE
os.environ["DEMO_MODE"] = "1"

# Load environment variables
load_dotenv()

# Global variables corresponding to main.py globals
# Global variables corresponding to main.py globals
_hydra_context = None
_preload_cache = {
    "cfg": None,
    "main_agent_tool_manager": None,
    "sub_agent_tool_managers": None,
    "output_formatter": None,
    "tool_definitions": None,
    "sub_agent_tool_definitions": None,
    "loaded": False,
}
_preload_lock = threading.Lock()

def load_miroflow_config(config_overrides: Optional[dict] = None, llm_config_name: Optional[str] = None) -> DictConfig:
    """
    Load the full MiroFlow configuration using Hydra.
    Adapted from main.py
    """

    global _hydra_initialized # Deprecated but harmless to keep if needed, but we removed it? No, wait. 
    # Let's just put global _hydra_context here
    
    global _hydra_context

    # Get the path to the miroflow agent config directory
    # apps/gradio-demo/flask_demo/app.py -> apps/miroflow-agent/conf ?
    # main.py was in apps/gradio-demo and looked for ../miroflow-agent/conf
    # So we are apps/gradio-demo/flask_demo
    # We need ../../miroflow-agent/conf relative to flask_demo
    # OR ../miroflow-agent/conf relative to gradio-demo
    
    # Path(__file__) is apps/gradio-demo/flask_demo/app.py
    # .parent is flask_demo
    # .parent.parent is gradio-demo
    # .parent.parent.parent is apps
    
    # main.py logic: Path(__file__).parent.parent / "miroflow-agent" / "conf"
    # if main.py is in apps/gradio-demo, .parent is gradio-demo, .parent.parent is apps.
    
    # So we need apps/miroflow-agent/conf
    # apps/flask_demo/app.py -> apps/miroflow-agent/conf
    # Path(__file__) is apps/flask_demo/app.py
    # .parent is flask_demo
    # .parent.parent is apps
    
    miroflow_config_dir = Path(__file__).parent.parent / "miroflow-agent" / "conf"
    miroflow_config_dir = miroflow_config_dir.resolve()
    logger.debug(f"Config dir: {miroflow_config_dir}")

    if not miroflow_config_dir.exists():
        # Fallback to main.py logic if structure is different
        # Try finding it relative to current working directory if run from gradio-demo
        miroflow_config_dir = Path("..") / "miroflow-agent" / "conf"
        miroflow_config_dir = miroflow_config_dir.resolve()
        if not miroflow_config_dir.exists():
             raise FileNotFoundError(
                f"MiroFlow config directory not found: {miroflow_config_dir}"
            )

    # Initialize Hydra globally ONCE.
    # We rely on GlobalHydra to check if it's initialized.
    # Since we are in a server environment, we want to initialize it and keep it valid.
    # The 'initialize_config_dir' returns a ContextManager. If we verify it's not initialized,
    # we enter it manually and DO NOT exit, effectively keeping it initialized globally.
    if not GlobalHydra.instance().is_initialized():
        try:
            # We need to use a relative path for initialize_config_dir or ensure absolute works correctly with Hydra
            # Store the context to prevent garbage collection (though Hydra global state handles it usually)
            global _hydra_context
            _hydra_context = initialize_config_dir(
                config_dir=str(miroflow_config_dir), version_base=None
            )
            _hydra_context.__enter__()
            logger.info(f"Hydra initialized globally at {miroflow_config_dir}")
        except Exception as e:
            # Check race condition again
            if GlobalHydra.instance().is_initialized():
                 logger.warning(f"Hydra initialized concurrently: {e}")
            else:
                 logger.error(f"Failed to initialize Hydra: {e}")
                 raise e

    # Compose configuration
    overrides = []
    
    # Defaults
    llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "qwen")
    model_name = os.getenv("DEFAULT_MODEL_NAME", "MiroThinker")
    agent_set = os.getenv("DEFAULT_AGENT_SET", "demo")
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")

    provider_config_map = {
        "anthropic": "claude-3-7",
        "openai": "gpt-5",
        "qwen": "qwen-3",
    }

    # Determine LLM config file to use
    if llm_config_name:
        llm_config = llm_config_name
    else:
        llm_config = provider_config_map.get(llm_provider, "qwen-3")
        
    overrides.extend([
        f"llm={llm_config}",
    ])
    
    if not llm_config_name:
        overrides.extend([
            f"llm.provider={llm_provider}",
            f"llm.model_name={model_name}",
        ])
        if base_url:
            overrides.append(f"llm.base_url={base_url}")
        if api_key:
            overrides.append(f"llm.api_key={api_key}")
    
    overrides.extend([
        f"agent={agent_set}",
        "agent.main_agent.max_turns=50",
        "benchmark=gaia-validation",
    ])

    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    overrides.append(f"{key}.{subkey}={subvalue}")
            else:
                overrides.append(f"{key}={value}")

    try:
        cfg = compose(config_name="config", overrides=overrides)
        return cfg
    except AssertionError as e:
        if "GlobalHydra is not initialized" in str(e):
            logger.warning("GlobalHydra appeared initialized but compose failed. Forcing re-initialization.")
            GlobalHydra.instance().clear()
            # Re-run initialization logic manually here to be safe
            _hydra_context = initialize_config_dir(
                config_dir=str(miroflow_config_dir), version_base=None
            )
            _hydra_context.__enter__()
            logger.info("Hydra forced re-initialized.")
            
            # Retry compose
            cfg = compose(config_name="config", overrides=overrides)
            return cfg
        else:
            raise e
    except Exception as e:
        logger.error(f"Failed to compose Hydra config: {e}")
        raise e

def _ensure_preloaded(config_overrides=None, llm_config_name=None):
    """Lazy load pipeline components."""
    global _preload_cache
    if _preload_cache["loaded"] and not config_overrides:
        return

    with _preload_lock:
        if _preload_cache["loaded"] and not config_overrides:
            return

        logger.info("Loading pipeline components...")
        cfg = load_miroflow_config(config_overrides, llm_config_name)
        main_agent_tool_manager, sub_agent_tool_managers, output_formatter = (
            create_pipeline_components(cfg)
        )
        tool_definitions = asyncio.run(
            main_agent_tool_manager.get_all_tool_definitions()
        )
        if cfg.agent.sub_agents:
            tool_definitions += expose_sub_agents_as_tools(cfg.agent.sub_agents)

        sub_agent_tool_definitions = {
            name: asyncio.run(sub_agent_tool_manager.get_all_tool_definitions())
            for name, sub_agent_tool_manager in sub_agent_tool_managers.items()
        }

        # If overwriting, just update cache
        _preload_cache["cfg"] = cfg
        _preload_cache["main_agent_tool_manager"] = main_agent_tool_manager
        _preload_cache["sub_agent_tool_managers"] = sub_agent_tool_managers
        _preload_cache["output_formatter"] = output_formatter
        _preload_cache["tool_definitions"] = tool_definitions
        _preload_cache["sub_agent_tool_definitions"] = sub_agent_tool_definitions
        _preload_cache["loaded"] = True
        logger.info("Pipeline components loaded successfully.")



# Helper functions from main.py
def filter_google_search_organic(organic: List[dict]) -> List[dict]:
    result = []
    print(f"organic------------------------------------------{organic}")

    for item in organic:
        result.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "date": item.get("date", ""),
            "source": item.get("source", "")
        })
    return result

def is_scrape_error(result: str) -> bool:
    try:
        json.loads(result)
        return False
    except json.JSONDecodeError:
        return True

def filter_message(message: dict) -> dict:
    if message["event"] == "tool_call":
        tool_name = message["data"].get("tool_name")
        tool_input = message["data"].get("tool_input")
        print(f"DEBUG_FILTER_MESSAGE: Processing tool_call for {tool_name}")
        if tool_name == "google_search" and isinstance(tool_input, dict) and "result" in tool_input:
            result_dict = json.loads(tool_input["result"])
            if "organic" in result_dict:
                new_result = {"organic": filter_google_search_organic(result_dict["organic"])}
                message["data"]["tool_input"]["result"] = json.dumps(new_result, ensure_ascii=False)
        if tool_name in ["scrape", "scrape_website", "reading", "scrape_and_extract_info"] and isinstance(tool_input, dict) and "result" in tool_input:
            if is_scrape_error(tool_input["result"]):
                message["data"]["tool_input"] = {"error": tool_input["result"]}
            else:
                 # Remove 'result' (too large) but keep 'url' and other args for UI
                safe_input = tool_input.copy()
                safe_input.pop("result", None)
                message["data"]["tool_input"] = safe_input
    return message


def events_generator(query: str, history=None):
    """Sync generator to yield SSE events"""
    
    # Format history into the query if provided
    final_task_description = query
    if history and len(history) > 1: # >1 because the last one is the current query which we already have
        context_str = "\n\n[Previous Conversation Context]\n"
        # Skip the last item if it's the current query (frontend pushes it before sending)
        effective_history = history[:-1] if history[-1]['role'] == 'user' and history[-1]['content'] == query else history
        
        for msg in effective_history:
            role = "User" if msg['role'] == 'user' else "AI"
            content = msg.get('content', '')
            context_str += f"{role}: {content}\n"
        
        context_str += "\n[Current Request]\n"
        final_task_description = context_str + query
        logger.info(f"Task with History: {final_task_description[:200]}...")

    task_id = str(uuid.uuid4())
    logger.info(f"Starting task {task_id} for query: {query}")
    
    stream_queue = queue.Queue()
    # Initialize Hydra globally ONCE.
    # We rely on GlobalHydra to check if it's initialized.
    # Since we are in a server environment, we want to initialize it and keep it valid.
    # The 'initialize_config_dir' returns a ContextManager. If we verify it's not initialized,
    # we enter it manually and DO NOT exit, effectively keeping it initialized globally.
    if not GlobalHydra.instance().is_initialized():
        try:
            # We need to use a relative path for initialize_config_dir or ensure absolute works correctly with Hydra
            # Store the context to prevent garbage collection (though Hydra global state handles it usually)
            global _hydra_context
            _hydra_context = initialize_config_dir(
                config_dir=str(miroflow_config_dir), version_base=None
            )
            _hydra_context.__enter__()
            logger.info(f"Hydra initialized globally at {miroflow_config_dir}")
        except Exception as e:
            # Check race condition again
            if GlobalHydra.instance().is_initialized():
                 logger.warning(f"Hydra initialized concurrently: {e}")
            else:
                 logger.error(f"Failed to initialize Hydra: {e}")
                 raise e
    cancel_event = threading.Event()

    def run_pipeline_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            class ThreadQueueWrapper:
                def __init__(self, thread_queue, cancel_event):
                    self.thread_queue = thread_queue
                    self.cancel_event = cancel_event

                async def put(self, item):
                    if self.cancel_event.is_set():
                        return
                    # Put directly into sync queue
                    self.thread_queue.put(filter_message(item))

            wrapper_queue = ThreadQueueWrapper(stream_queue, cancel_event)
            
            # Ensure loaded
            _ensure_preloaded()

            loop.run_until_complete(execute_task_pipeline(
                cfg=_preload_cache["cfg"],
                task_id=task_id,
                task_description=final_task_description,
                task_file_name=None,
                main_agent_tool_manager=_preload_cache["main_agent_tool_manager"],
                sub_agent_tool_managers=_preload_cache["sub_agent_tool_managers"],
                output_formatter=_preload_cache["output_formatter"],
                stream_queue=wrapper_queue,
                log_dir=os.getenv("LOG_DIR", "logs/api-server"),
                tool_definitions=_preload_cache["tool_definitions"],
                sub_agent_tool_definitions=_preload_cache["sub_agent_tool_definitions"],
            ))
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            stream_queue.put({
                "event": "error",
                "data": {"error": str(e)}
            })
        finally:
            stream_queue.put(None)  # Sentinel
            if "loop" in locals():
                loop.close()

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(run_pipeline_in_thread)

    while True:
        try:
            # Blocking get with timeout to allow checking for other conditions if needed
            # or just simple get()
            message = stream_queue.get(timeout=1.0)
            if message is None:
                break
            
            # Apply message filtering/transformation (e.g. search result processing)
            message = filter_message(message)

            # Format as SSE
            yield f"data: {json.dumps(message)}\n\n"
        except queue.Empty:
             yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"
            break
            
    executor.shutdown(wait=False)

import uuid

# Flask wrappers for async
def boot_app():
    # Only try to preload partially, real load happens on first request or here
    # We defer loading to avoid startup delay or config issues until needed
    pass


def get_available_agents():
    """List available agent configurations from the conf/agent directory."""
    try:
        # Use existing logic to find the config dir
        # Re-use miroflow_config_dir logic if possible, or recalculate
        miroflow_config_dir = Path(__file__).parent.parent / "miroflow-agent" / "conf"
        agent_dir = miroflow_config_dir / "agent"
        
        if not agent_dir.exists():
            return []
            
        agents = []
        for file_path in agent_dir.glob("*.yaml"):
            # Use filename stem as ID
            agent_id = file_path.stem
            # Create a display name (capitalize, replace underscores)
            display_name = agent_id.replace("_", " ").title()
            
            # Special handling for known agents to have nicer names if desired
            if agent_id == "demo": display_name = "Demo Agent"
            elif agent_id == "browsing": display_name = "Browsing Agent" # example
            
            agents.append({"id": agent_id, "name": display_name})
            
        # Sort agents, maybe put 'demo' first
        agents.sort(key=lambda x: (x['id'] != 'demo', x['id']))
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        return []

# Chat History Storage
CHAT_HISTORY_DIR = current_dir / "chat_history"
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    try:
        sessions = []
        for file_path in CHAT_HISTORY_DIR.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        "id": file_path.stem,
                        "title": data.get("title", "New Chat"),
                        "timestamp": data.get("timestamp", int(os.path.getmtime(file_path) * 1000))
                    })
            except Exception as e:
                logger.error(f"Error reading session {file_path}: {e}")
        
        # Sort by timestamp desc
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify(sessions)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    try:
        file_path = CHAT_HISTORY_DIR / f"{session_id}.json"
        if not file_path.exists():
            return jsonify({"error": "Session not found"}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['POST'])
def save_session(session_id):
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        file_path = CHAT_HISTORY_DIR / f"{session_id}.json"
        
        # Ensure timestamp
        if "timestamp" not in data:
            data["timestamp"] = int(time.time() * 1000)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        file_path = CHAT_HISTORY_DIR / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500

def get_available_models():
    """Scan conf/llm directory for available model configurations."""
    models = []
    llm_conf_dir = Path(__file__).parent.parent / "miroflow-agent" / "conf" / "llm"
    
    # Defaults if directory lookup fails
    defaults = [
        {"id": "MiroThinker", "name": "MiroThinker (Default)"},
        {"id": "gpt-4o", "name": "GPT-4o"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
        {"id": "doubao-seed-1-6-251015", "name": "Doubao Pro (1.6)"},
        {"id": "qwen-2.5-72b-instruct", "name": "Qwen 2.5 72B"},
    ]
    
    try:
        if llm_conf_dir.exists():
            for file_path in llm_conf_dir.glob("*.yaml"):
                if file_path.stem == "default":
                    continue
                # Use filename as ID
                models.append({
                    "id": file_path.stem, 
                    "name": file_path.stem.replace("-", " ").title()
                })
            
            # Sort models
            models.sort(key=lambda x: x['name'])
            
            # Add defaults if not present (or we can just rely on files)
            # For now, let's prepend the 'Default' (which maps to default.yaml logic)
            models.insert(0, {"id": "default", "name": "Default (Configured)"})
            return models
        return defaults
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return defaults

@app.route('/')
def index():
    agents = get_available_agents()
    models = get_available_models()
    return render_template('index.html', agents=agents, models=models)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    # If session_id is provided, we could use it for server-side state if needed, 
    # but for now we rely on client-side full history + file storage.
    
    model = data.get('model', 'MiroThinker')
    agent = data.get('agent', 'demo')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Update config if needed (simple hack - in real app, better to handle per request config)
    # Since Hydra is global, switching models globally is tricky without reloading.
    # For this demo, we can just try to reload config if model changed, 
    # but concurrent requests would conflict.
    # We will assume single user for demo or accept race info.
    
    # Simplest: just use current config for now, or reload if we allow switching.
    # Let's support model switching by reloading config if different.
    
    current_model = None
    if _preload_cache["cfg"]:
        current_model = _preload_cache["cfg"].llm.model_name
        
    if current_model != model:
        # Requesting a different model, need to reload. 
        # Note: This is not thread safe for multiple users!
        logger.info(f"Switching model from {current_model} to {model}")
        
        # Try to match model ID to a config file
        llm_conf_dir = Path(__file__).parent.parent / "miroflow-agent" / "conf" / "llm"
        potential_config = llm_conf_dir / f"{model}.yaml"
        
        provider = "openai" # Default safe fallback
        llm_config_name = None

        if potential_config.exists():
             llm_config_name = model
             # We assume the YAML file contains the provider info, but we need to pass a valid provider 
             # to the hydration logic if it's strict. 
             # However, typically 'llm_config_name' dictates which file is loaded.
             # We set provider to generic 'qwen' or 'openai' just to satisfy the dict, 
             # but the loaded config should override it.
             logger.info(f"Found specific config file for model: {model}")
        elif "gpt-4o-mini" in model:
            provider = "openai"
            llm_config_name = "gpt-4o-mini"
        elif "gpt" in model or "o1" in model:
            provider = "openai"
            llm_config_name = "gpt-5" # Fallback or default
        elif "claude" in model:
            provider = "anthropic"
            llm_config_name = "claude-3-7"
        elif "doubao" in model:
            provider = "doubao"
            llm_config_name = "doubao"
        elif "qwen" in model:
             provider = "qwen"
             llm_config_name = "qwen-3" # Fallback if specific one not found
        else:
             # Default fallback if no file and no match
             llm_config_name = "default"

        # We pass llm_config_name to effectively switch the base config file.
        # We still pass config_overrides just in case, but rely on config file for base_url
        config_overrides = {
            "llm": {
                "model_name": model, # This might override what's in the file, be careful.
                "provider": provider
            },
            "agent": agent
        }
        
        # Special handling for Doubao:
        # The user sends "doubao" but the real model ID is in the yaml (e.g. doubao-seed-...)
        if "doubao" in model and not potential_config.exists():
             # ... existing doubao logic ...
             pass
             
        # If we found a specific config file, we should arguably NOT override model_name 
        # because the file might have the specific technical model name (like .gguf)
        if potential_config.exists():
            # Don't override model_name if we are loading a full config file for it
            if "model_name" in config_overrides["llm"]:
                del config_overrides["llm"]["model_name"]
            # Don't override provider if we don't know it, let config decide
            if "provider" in config_overrides["llm"]:
                 del config_overrides["llm"]["provider"]
        # We should NOT override model_name with "doubao" alias.
        if "doubao" in model:
                del config_overrides["llm"]["model_name"]

        # In this simple demo, we force reload
        _preload_cache["loaded"] = False
        _ensure_preloaded(config_overrides, llm_config_name)
    else:
        _ensure_preloaded()

    history = data.get('history', [])

    return Response(stream_with_context(events_generator(query, history)), 
                    mimetype='text/event-stream')

def extract_query_from_messages(messages):
    """
    Extract the last user message to serve as the agent query.
    Basic implementation: takes the content of the last message where role='user'.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None

import time

def openai_stream_generator(query: str, model: str):
    """
    Generator that adapts MiroFlow agent events into OpenAI-compatible SSE chunks.
    """
    final_task_description = query
    # Reuse common logic for starting pipeline
    # We duplicate some setup logic here or we could refactor 'chat' to separate setup from response
    # For safety/speed, I'll copy the setup pattern but assume config is loaded/switched by the caller route
    
    task_id = str(uuid.uuid4())
    logger.info(f"Starting API task {task_id} for query: {query}")
    
    stream_queue = queue.Queue()
    
    # Ensure Hydra context (same pattern as events_generator/chat)
    if not GlobalHydra.instance().is_initialized():
         # In theory request handler checked this, but safety first
         pass 

    cancel_event = threading.Event()

    def run_pipeline_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Re-define wrapper here or make it a global helper if used twice
            class ThreadQueueWrapper:
                def __init__(self, thread_queue, cancel_event):
                    self.thread_queue = thread_queue
                    self.cancel_event = cancel_event

                async def put(self, item):
                    if self.cancel_event.is_set():
                        return
                    self.thread_queue.put(filter_message(item))

            wrapper_queue = ThreadQueueWrapper(stream_queue, cancel_event)
            _ensure_preloaded() # Should be loaded by route

            result = loop.run_until_complete(execute_task_pipeline(
                cfg=_preload_cache["cfg"],
                task_id=task_id,
                task_description=final_task_description,
                task_file_name=None,
                main_agent_tool_manager=_preload_cache["main_agent_tool_manager"],
                sub_agent_tool_managers=_preload_cache["sub_agent_tool_managers"],
                output_formatter=_preload_cache["output_formatter"],
                stream_queue=wrapper_queue,
                log_dir=os.getenv("LOG_DIR", "logs/api-server"),
                tool_definitions=_preload_cache["tool_definitions"],
                sub_agent_tool_definitions=_preload_cache["sub_agent_tool_definitions"],
            ))
            # Stream the final result explicitly
            stream_queue.put({"event": "final_response", "data": result})
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            stream_queue.put({"event": "error", "data": {"error": str(e)}})
        finally:
            stream_queue.put(None)
            if "loop" in locals():
                loop.close()

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(run_pipeline_in_thread)
    
    created_time = int(time.time())
    
    while True:
        try:
            message = stream_queue.get(timeout=1.0)
            if message is None:
                break
            
            # Map MiroFlow events to OpenAI chunks
            # Event structure: {"event": "...", "data": ...} or similar from filter_message
            # Filter message limits keys. 
            # We assume message is one of: tool_call, error, step?
            # Actually filter_message returns the dict directly.
            
            # For a pure chat interface, we might only care about the final answer or interim text?
            # Using 'step' logs as 'content' might be noisy. 
            # Ideally the agent yields chunks of the final answer.
            # But currently MiroFlow agent seems to yield "steps" (logs) and maybe final result?
            # Let's stream everything as content for now so the user sees *something*.
            
            content = None
            if isinstance(message, dict):
                # Try to extract readable text
                if "data" in message:
                    data = message["data"]
                    if isinstance(data, dict):
                         # Maybe tool input/output logs?
                         if "tool_name" in data:
                             content = f"[Tool: {data['tool_name']}] "
                         elif "error" in data:
                             content = f"[Error: {data['error']}] "
                         else:
                             content = str(data)
                    else:
                         content = str(data)
                elif "event" in message:
                     if message['event'] == 'final_response':
                         # This is the actual answer
                         content = str(message['data'])
                     else:
                         content = f"[{message['event']}] "
            
            if content:
                chunk = {
                    "id": f"chatcmpl-{task_id}",
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content + "\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        except queue.Empty:
            # Heartbeat (optional, or send empty comment)
            pass
        except Exception as e:
            chunk = {
                    "id": f"chatcmpl-{task_id}",
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"[System Error: {str(e)}]"},
                        "finish_reason": "stop"
                    }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            break
            
    # Done
    yield "data: [DONE]\n\n"
    executor.shutdown(wait=False)

@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    """
    OpenAI-compatible endpoint for Chat Completions.
    Supports 'model' (Agent Name/LLM Model), 'messages', and 'stream'.
    Maps Agent Thoughts/Tools to 'reasoning_content' (DeepSeek R1 style).
    """
    data = request.json or {}
    messages = data.get('messages', [])
    requested_model = data.get('model', 'news_monitor')
    stream = data.get('stream', False)

    # 1. Parse Last User Message
    user_query = "Hello"
    if messages:
        # Simple extraction of last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

    # 2. Parse Agent and Model
    # Format: "agent_name" OR "agent_name/llm_model"
    parse_result = requested_model.split('/', 1)
    target_agent = parse_result[0]
    target_llm = parse_result[1] if len(parse_result) > 1 else None
    
    # Logic to switch LLM config if requested
    if target_llm:
        llm_config_name = "llm/openai" # default
        provider = "openai"
        
        if "doubao" in target_llm:
            llm_config_name = "llm/doubao"
            provider = "doubao"
        elif "qwen" in target_llm:
            llm_config_name = "llm/qwen"
            provider = "dashscope"
        elif "claude" in target_llm:
            llm_config_name = "llm/anthropic"
            provider = "anthropic"
        elif "gemini" in target_llm:
            llm_config_name = "llm/gemini"
            provider = "google"
            
        config_overrides = {
            "llm": {
                "model_name": target_llm,
                "provider": provider
            }
        }
        
        # Doubao specific: Don't override model_name with generic alias if config handles it
        if "doubao" in target_llm:
             del config_overrides["llm"]["model_name"]

        _preload_cache["loaded"] = False
        _ensure_preloaded(config_overrides, llm_config_name)
    else:
        # Just ensure default loaded
        _ensure_preloaded()

    # 3. Check Agent Existence
    logging.info(f"OpenAI API Request: Agent={target_agent}, Model={target_llm or 'default'}, Query={user_query[:50]}...")
    
    try:
        agent = get_agent_by_name(target_agent)
    except Exception as e:
        return jsonify({"error": {"message": f"Agent '{target_agent}' not found. Usage: 'agent_name' or 'agent_name/model_name'", "type": "invalid_request_error"}}), 404

    # 4. Stream Generator
    def generate_openai_stream():
        chat_id = "chatcmpl-" + str(uuid.uuid4())
        created_time = int(time.time())

        # Helper to yield chunk
        def yield_chunk(delta_dict, finish_reason=None):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": requested_model,
                "choices": [{
                    "index": 0,
                    "delta": delta_dict,
                    "finish_reason": finish_reason
                }]
            }
            return f"data: {json.dumps(chunk)}\n\n"

        # Yield Initial Role
        yield yield_chunk({"role": "assistant"})

        # Run Agent
        try:
            event_queue = queue.Queue()
            
            def run_sync():
                try:
                    setup_miroflow_env()
                    asyncio.run(agent.run(
                        task=user_query,
                        output_queue=event_queue
                    ))
                    event_queue.put(None) # Sentinel
                except Exception as e:
                    logging.error(f"Agent Run Error: {e}")
                    event_queue.put({"event": "error", "error": str(e)})

            t = threading.Thread(target=run_sync)
            t.start()

            # Consumer Loop
            while True:
                try:
                    event_data = event_queue.get(timeout=1.0)
                except queue.Empty:
                    if not t.is_alive():
                        break
                    continue

                if event_data is None: 
                    break # Done
                
                if isinstance(event_data, dict) and event_data.get("event") == "error":
                     yield yield_chunk({"content": f"\n\n[Error: {event_data.get('error')}]"}, "stop")
                     break

                # Parse Event (Standard MiroFlow Event Object)
                event_type = getattr(event_data, 'event_type', None)
                
                if not event_type: 
                    continue

                if event_type == 'agent_thought':
                    thought = getattr(event_data, 'thought', "") or getattr(event_data, 'content', "")
                    if thought:
                        yield yield_chunk({"reasoning_content": thought})

                elif event_type == 'tool_call':
                    tool_name = getattr(event_data, 'tool_name', 'tool')
                    tool_input = getattr(event_data, 'tool_input', {})
                    log = f"\n\n**Tool Call** (`{tool_name}`):\n```json\n{json.dumps(tool_input, ensure_ascii=False)}\n```\n\n"
                    yield yield_chunk({"reasoning_content": log})

                elif event_type == 'tool_result':
                    result = getattr(event_data, 'tool_result', '')
                    # Limit log size
                    res_str = str(result)
                    if len(res_str) > 1000: res_str = res_str[:1000] + "...(truncated)"
                    log = f"**Result**:\n```text\n{res_str}\n```\n"
                    yield yield_chunk({"reasoning_content": log})

                elif event_type == 'response_chunk':
                    content = getattr(event_data, 'content', "")
                    if content:
                        yield yield_chunk({"content": content})

            # Done
            yield yield_chunk({}, "stop")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Streaming Error: {e}")
            yield yield_chunk({"content": f"\n\n[System Error: {str(e)}]"}, "stop")

    if stream:
        return Response(stream_with_context(generate_openai_stream()), mimetype='text/event-stream')
    else:
        # Sync Fallback (Not implemented fully, returning error to force stream)
        return jsonify({
            "error": {
                "message": "Please use 'stream': true. Non-streaming is not supported in this demo.",
                "type": "invalid_request_error"
            }
        }), 400



if __name__ == '__main__':
    # Ensure event loop policy for MacOS/Unix
    if sys.platform.lower().startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Load default configs? No, lazy load on request
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
