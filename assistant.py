import subprocess
import json
import os
import signal
import logging
import platform
from datetime import datetime
from difflib import SequenceMatcher
from contextlib import contextmanager
from dotenv import load_dotenv
import ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Configuration
KB_FILE = "knowledge_base.json"
LOG_FILE = "cmd_ai_assistant.log"
DANGEROUS_COMMANDS = ["format", "del /s", "rd /s", "rmdir /s", "deltree"]
LLM_TIMEOUT = 30  # seconds

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Command history
command_history = []


@contextmanager
def timeout(duration):
    """Context manager for timeout operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def get_system_context():
    """Get system information for better error context."""
    return {
        "os": platform.system(),
        "version": platform.version(),
        "cwd": os.getcwd()
    }


def load_kb():
    """Load knowledge base from JSON file."""
    if not os.path.exists(KB_FILE):
        return []
    try:
        with open(KB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error("Failed to load knowledge base - corrupted JSON")
        return []


def save_kb(kb):
    """Save knowledge base to JSON file."""
    try:
        with open(KB_FILE, "w") as f:
            json.dump(kb, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to save knowledge base: {e}")


def add_to_kb(command, error, solution, model_used="local"):
    """Add a new entry to the knowledge base."""
    kb = load_kb()
    kb.append({
        "command": command,
        "error": error,
        "solution": solution,
        "timestamp": datetime.now().isoformat(),
        "model": model_used,
        "success_count": 0
    })
    save_kb(kb)
    logging.info(f"Added to KB: {command[:50]}... | Model: {model_used}")


def check_kb_for_match(command, error, threshold=0.7):
    """Check knowledge base for similar errors using fuzzy matching."""
    kb = load_kb()
    best_match = None
    best_score = 0
    
    for item in kb:
        # Calculate similarity for both command and error
        cmd_similarity = SequenceMatcher(None, item["command"], command).ratio()
        err_similarity = SequenceMatcher(None, item["error"], error).ratio()
        
        combined_score = (cmd_similarity + err_similarity) / 2
        
        if combined_score > best_score and combined_score > threshold:
            best_score = combined_score
            best_match = item
    
    if best_match:
        # Update success count
        best_match["success_count"] = best_match.get("success_count", 0) + 1
        save_kb(kb)
        return best_match["solution"]
    
    return None


def is_safe_command(cmd):
    """Check if command is potentially dangerous."""
    cmd_lower = cmd.lower()
    for danger in DANGEROUS_COMMANDS:
        if danger in cmd_lower:
            return False
    return True


def local_llm_fix(command, error):
    """Attempt to fix command using local Ollama model."""
    ctx = get_system_context()
    prompt = f"""
You are an AI that helps fix Windows CMD commands.
System: {ctx['os']} {ctx['version']}
Working Directory: {ctx['cwd']}

Command:
{command}

Error:
{error}

Explain why it happened and propose a corrected command.
The response should include:
1) Why it failed
2) The correct command
3) Extra tips if needed
"""
    try:
        with timeout(LLM_TIMEOUT):
            response = ollama.generate(model="phi3", prompt=prompt)
            return response['response']
    except TimeoutError:
        logging.warning("Local LLM timeout")
        return None
    except Exception as e:
        logging.error(f"Local LLM error: {e}")
        return None


def gemini_fix(command, error, additional_context=""):
    """Attempt to fix command using Gemini."""
    ctx = get_system_context()
    prompt = f"""
You are a highly skilled Windows command-line troubleshooting assistant.
System: {ctx['os']} {ctx['version']}
Working Directory: {ctx['cwd']}

A user ran this command and it failed.

Command:
{command}

Error:
{error}

{f"User feedback: {additional_context}" if additional_context else ""}

Explain why it happened clearly and propose a corrected command.
Include:
1. Explanation
2. Corrected command
3. Extra guidance if helpful
"""
    try:
        result = gemini_llm.invoke([HumanMessage(content=prompt)])
        return result.content
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return f"Error contacting Gemini API: {e}"


def run_cmd(command):
    """Execute a command and return success status and output."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if completed.returncode == 0:
            print(completed.stdout)
            return True, completed.stdout
        else:
            return False, completed.stderr
    except subprocess.TimeoutExpired:
        return False, "Command execution timed out"
    except Exception as e:
        return False, str(e)


def show_history():
    """Display command history."""
    if not command_history:
        print("No command history yet.")
        return
    
    print("\nCommand History:")
    print("-" * 50)
    for i, hist_cmd in enumerate(command_history, 1):
        print(f"{i}. {hist_cmd}")
    print("-" * 50)


def get_command_from_history(reference):
    """Get command from history by reference number."""
    try:
        idx = int(reference[1:]) - 1
        if 0 <= idx < len(command_history):
            return command_history[idx]
        else:
            print("Invalid history reference number.")
            return None
    except (ValueError, IndexError):
        print("Invalid history reference format. Use !<number>")
        return None


def main():
    """Main application loop."""
    print("AI CMD Assistant for Windows 11 (Ollama phi3 + Gemini)")
    print("Verbose Troubleshooting Mode Enabled")
    print("=" * 60)
    print("Commands:")
    print("  history    - Show command history")
    print("  !<number>  - Run command from history")
    print("  exit/quit  - Exit the assistant")
    print("=" * 60)
    print()
    
    while True:
        try:
            cmd = input("cmd-ai> ").strip()
            
            # Handle special commands
            if cmd.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if cmd == "history":
                show_history()
                continue
            
            if cmd.startswith("!"):
                cmd = get_command_from_history(cmd)
                if cmd is None:
                    continue
                print(f"Running: {cmd}")
            
            if not cmd:
                continue
            
            # Safety check
            if not is_safe_command(cmd):
                confirm = input("Potentially dangerous command detected. Continue? (yes/no): ").strip().lower()
                if confirm != "yes":
                    print("Command cancelled.")
                    continue
            
            # Add to history
            command_history.append(cmd)
            
            # Execute command
            success, output = run_cmd(cmd)
            logging.info(f"Command: {cmd} | Success: {success}")
            
            if success:
                continue
            
            # Command failed - start troubleshooting
            print("\nCommand failed.")
            print(f"Error:\n{output}")
            
            # Check knowledge base first
            kb_suggestion = check_kb_for_match(cmd, output)
            if kb_suggestion:
                print("\nSimilar issue found in local knowledge base!")
                print(f"Suggested Fix:\n{kb_suggestion}")
                continue
            
            # Try local LLM
            print("\nAttempting local fix using phi3...")
            local_fix = local_llm_fix(cmd, output)
            
            if local_fix is None:
                print("Local model unavailable, using Gemini directly...")
                gemini_solution = gemini_fix(cmd, output)
                print(f"\nGemini Suggestion:\n{gemini_solution}")
                add_to_kb(cmd, output, gemini_solution, model_used="gemini")
                continue
            
            print(f"\nSuggested Fix (Local):\n{local_fix}")
            
            # Get user feedback
            user_input = input("\nDid this fix help? (y/n): ").strip().lower()
            if user_input == "y":
                add_to_kb(cmd, output, local_fix, model_used="local")
                continue
            
            # Escalate to Gemini with optional context
            additional_context = input("What went wrong? (or press Enter to skip): ").strip()
            
            print("\nEscalating to Gemini...")
            gemini_solution = gemini_fix(cmd, output, additional_context)
            print(f"\nGemini Suggestion:\n{gemini_solution}")
            add_to_kb(cmd, output, gemini_solution, model_used="gemini")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()