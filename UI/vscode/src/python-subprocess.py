#!/usr/bin/env python3
"""
Python subprocess for VSCode Kai Agent extension.
"""
import sys
import json
import asyncio
import logging
import os
import time
from pathlib import Path
from datetime import datetime

print("PYTHON SCRIPT EXECUTING!", file=sys.stderr)
print(f"Python executable: {sys.executable}", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Current directory: {os.getcwd()}", file=sys.stderr)
print(f"Python path entries (first 5):", file=sys.stderr)
for i, p in enumerate(sys.path[:5]):
    print(f"  [{i}] {p}", file=sys.stderr)

# Import kai configuration to get paths
try:
    from kai.config.paths import AGENT_BASE_DIR
    print(f"✅ Successfully imported AGENT_BASE_DIR: {AGENT_BASE_DIR}", file=sys.stderr)
except Exception as e:
    print(f"❌ Failed to import AGENT_BASE_DIR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    # Fallback to home directory
    AGENT_BASE_DIR = Path.home() / '.kai_agent'
    print(f"Using fallback AGENT_BASE_DIR: {AGENT_BASE_DIR}", file=sys.stderr)

# Set up debug logging to file using kai's configured paths
debug_log_dir = AGENT_BASE_DIR / 'debug_logs'
debug_log_dir.mkdir(parents=True, exist_ok=True)
debug_log_file = debug_log_dir / f'vscode_subprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

def log_error_to_file(error_msg, full_traceback):
    """Log detailed error information to debug file."""
    try:
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR TIMESTAMP: {datetime.now().isoformat()}\n")
            f.write(f"ERROR MESSAGE: {error_msg}\n")
            f.write(f"FULL TRACEBACK:\n{full_traceback}\n")
            f.write(f"{'='*80}\n\n")
    except Exception as log_error:
        print(f"Failed to log error to file: {log_error}", file=sys.stderr)

from kai.core.agent import KaiAgent
from kai.core.orchestration.ui_communicator import UICommunicator

# Enable VSCode mode for console messages (JSON to stdout instead of logger)
UICommunicator.set_vscode_mode(True)

# Disable telemetry and analytics
os.environ['DISABLE_TELEMETRY'] = '1'
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['DO_NOT_TRACK'] = '1'
os.environ['DISABLE_ANALYTICS'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = '0'
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_TELEMETRY'] = 'false'

# Monkey patch to disable any remaining telemetry
import requests
original_post = requests.post
def no_telemetry_post(*args, **kwargs):
    url = args[0] if args else kwargs.get('url', '')
    if 'posthog' in url or 'telemetry' in url or 'analytics' in url:
        return None
    return original_post(*args, **kwargs)
requests.post = no_telemetry_post

# Redirect all logging to stderr so stdout only contains JSON
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

async def handle_message(agent, request_id, message, context, session_id):
    """Handle message and return complete response."""
    try:
        if context is None:
            context = {}
        context['request_id'] = request_id
        
        response, new_session_id = await agent.chat(user_input=message, session_id=session_id, user_id="anonymous", context=context)
        
        # Send complete response immediately
        print(json.dumps({
            "type": "response",
            "request_id": request_id,
            "response": response,
            "session_id": new_session_id,
            "timing": {"agent_complete": time.time() * 1000}
        }))
        sys.stdout.flush()
        
        return new_session_id
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        error_msg = f"Error in message handling: {str(e)}"

        # Log to debug file
        log_error_to_file(error_msg, full_traceback)

        # Send error response to VS Code
        print(json.dumps({"type": "error", "message": f"{error_msg}\n\nFull traceback:\n{full_traceback}", "request_id": request_id}))
        sys.stdout.flush()
        print(f"Error in handle_message: {full_traceback}", file=sys.stderr)
        return session_id


async def main():
    # Handle empty model parameter
    turbo = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else "false"
    api_key = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
    # Not currently communicated from vscode, keep for later:
    model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else None
    
    session_id = None
    agent = None
    agent_initializing = False
    agent_initialized = False
    
    # Send initialized immediately to prevent timeout
    print(json.dumps({"type": "initialized"}))
    sys.stdout.flush()
    
    async def initialize_agent():
        """Initialize agent in background."""
        nonlocal agent, agent_initialized, agent_initializing
        
        if agent_initialized or agent_initializing:
            return
        
        agent_initializing = True
        llm_provider = 'ollama-turbo' if turbo == "true" else 'ollama'
        
        try:
            agent = KaiAgent(llm_provider=llm_provider, model=model, api_key=api_key)
            agent_initialized = True
            print(json.dumps({"type": "status", "message": "Agent ready!"}))
            sys.stdout.flush()
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_msg = f"Agent initialization failed: {str(e)}"

            # Log to debug file
            log_error_to_file(error_msg, full_traceback)

            print(json.dumps({"type": "error", "message": error_msg}))
            sys.stdout.flush()
            print(json.dumps({
                "type": "debug",
                "message": f"Agent creation error traceback: {full_traceback}"
            }))
            sys.stdout.flush()
            agent_initializing = False  # Allow retry
    
    # Track active tasks for proper cleanup
    active_tasks = set()

    try:
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

                if not line:
                    break

                request = json.loads(line.strip())

                if request.get("type") == "chat":
                    # Ensure agent is initialized before handling message
                    if not agent_initialized and not agent_initializing:
                        await initialize_agent()

                    # Wait for initialization if in progress
                    while agent_initializing and not agent_initialized:
                        await asyncio.sleep(0.1)

                    if agent is None:
                        # Agent initialization failed - send error response
                        print(json.dumps({
                            "type": "error",
                            "message": "Agent failed to initialize. Please check the logs and restart the extension.",
                            "request_id": request["request_id"]
                        }))
                        sys.stdout.flush()
                    else:
                        # Handle the message (agent successfully initialized)
                        session_id = await handle_message(
                            agent,
                            request["request_id"],
                            request["message"],
                            request.get("context", {}),
                            session_id
                        )

                elif request.get("type") == "stop_autonomous":
                    # Wait for initialization if in progress (don't start new initialization)
                    while agent_initializing and not agent_initialized:
                        await asyncio.sleep(0.1)

                    if agent is not None:
                        result = agent.stop_autonomous_execution()
                    else:
                        result = "Agent not initialized - no autonomous session to stop."

                    print(json.dumps({"type": "stop_autonomous_response", "request_id": request["request_id"], "result": result}))
                    sys.stdout.flush()

                elif request.get("type") == "execution_progress_check":
                    # Handle execution progress monitoring as a concurrent task
                    # This allows monitoring to happen while autonomous execution is ongoing
                    async def handle_progress_check():
                        nonlocal agent, agent_initialized, agent_initializing

                        # Wait for initialization if in progress
                        while agent_initializing and not agent_initialized:
                            await asyncio.sleep(0.1)

                        if agent is not None:
                            context = request.get("context", {})
                            # Call the orchestrator's progress check handler
                            result = await agent.orchestrator._handle_execution_progress_check(
                                context=context,
                                session_metadata={}
                            )
                        else:
                            # Agent not initialized - default to continue
                            result = {"action": "continue", "feedback": "Agent not initialized, defaulting to continue"}

                        print(json.dumps({
                            "type": "execution_progress_check_response",
                            "request_id": request["request_id"],
                            "action": result.get("action", "continue"),
                            "feedback": result.get("feedback", "")
                        }))
                        sys.stdout.flush()

                    # Create task and track it
                    task = asyncio.create_task(handle_progress_check())
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)

            except json.JSONDecodeError as e:
                import traceback
                full_traceback = traceback.format_exc()
                error_msg = f"JSON decode error in main loop: {str(e)}"
                log_error_to_file(error_msg, full_traceback)
                print(f"JSON decode error: {error_msg}", file=sys.stderr)

            except Exception as e:
                import traceback
                full_traceback = traceback.format_exc()
                error_msg = f"Unexpected error in main loop: {str(e)}"
                log_error_to_file(error_msg, full_traceback)
                print(f"Unexpected error in main loop: {full_traceback}", file=sys.stderr)

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT", file=sys.stderr)
        sys.stderr.flush()
        pass
    
    print("MAIN FUNCTION ENDING", file=sys.stderr)

print("ABOUT TO CHECK __main__", file=sys.stderr)
if __name__ == "__main__":
    print("__main__ CHECK PASSED!", file=sys.stderr)
    print("CALLING ASYNCIO.RUN(MAIN())", file=sys.stderr)
    asyncio.run(main())
    print("ASYNCIO.RUN() COMPLETED!", file=sys.stderr)
else:
    print("__main__ CHECK FAILED - script not running as main", file=sys.stderr)