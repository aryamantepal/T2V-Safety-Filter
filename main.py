import requests
import json
from typing import List, Dict, Tuple
import time
import gzip
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "PKU-Alignment/SafeSora"
FILENAME = "config-train.json.gz"

# Download and read the gzipped JSON file
file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")

# Read the gzipped JSON file
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

# Convert to DataFrame if needed
rawDataset = pd.DataFrame(data)
dataset = pd.DataFrame(rawDataset.video_1[0]["prompt_text"] + rawDataset.prompt_type)


def read_file(file_path: str, default_value: str = "") -> str:
    """
    Read content from a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return default_value


print("\nDataset Info:")
print(rawDataset.columns)  # See what columns we have
print("\nUnique prompt types:")
print(rawDataset['prompt_type'].unique())  # See what values are in prompt_type


def read_prompts(temp_unsafe_folder: str, numTempUnsafe: int, numFrameUnsafe: int, numSafe: int) -> List[
    Tuple[str, str]]:
    prompts = []

    try:
        # Read temporal unsafe prompts from file
        with open(temp_unsafe_folder, 'r', encoding='utf-8') as f:
            temporal_prompts = [line.strip() for line in f.readlines() if line.strip()]
            prompts.extend((prompt, "temp_unsafe") for prompt in temporal_prompts[:numTempUnsafe])

        print("\nDebugging dataset access:")
        # Get safe prompts from dataset
        safe_count = 0
        safe_prompts = rawDataset[rawDataset['prompt_type'] == 'safety_neutral']
        print(f"Found {len(safe_prompts)} potential safe prompts")

        for _, row in safe_prompts.head(numSafe).iterrows():
            try:
                prompt = row['video_1']['prompt_text']
                prompts.append((prompt, "safe"))
                safe_count += 1
            except Exception as e:
                print(f"Error processing safe prompt: {str(e)}")

        # Get frame unsafe prompts from dataset
        frame_unsafe_count = 0
        unsafe_prompts = rawDataset[rawDataset['prompt_type'] == 'safety_critical']
        print(f"Found {len(unsafe_prompts)} potential unsafe prompts")

        for _, row in unsafe_prompts.head(numFrameUnsafe).iterrows():
            try:
                prompt = row['video_1']['prompt_text']
                prompts.append((prompt, "frame_unsafe"))
                frame_unsafe_count += 1
            except Exception as e:
                print(f"Error processing unsafe prompt: {str(e)}")

        print(f"\nLoaded {len([p for p in prompts if p[1] == 'temp_unsafe'])} temporal unsafe prompts")
        print(f"Loaded {len([p for p in prompts if p[1] == 'safe'])} safe prompts")
        print(f"Loaded {len([p for p in prompts if p[1] == 'frame_unsafe'])} frame unsafe prompts")

        return prompts

    except Exception as e:
        print(f"Error reading prompts: {str(e)}")
        return []

def reset_model(model_name: str) -> bool:
    """
    Reset the model's context.
    """
    try:
        # Create a new instance with a random context ID
        timestamp = str(time.time()).replace('.', '')
        context_id = f"ctx_{timestamp}"

        data = {
            "model": model_name,
            "context_id": context_id,
            "prompt": ""  # Empty prompt to initialize context
        }

        url = "http://localhost:11434/api/generate"
        response = requests.post(url, json=data)

        if response.status_code != 200:
            print(f"Error resetting model: Status code {response.status_code}")
            return False

        return True

    except Exception as e:
        print(f"Error resetting model: {str(e)}")
        return False


def model_run(model_name: str, numTempUnsafe: int, numFrameUnsafe: int, numSafe: int, output_file: str = "responses.json") -> None:
    """
    Read prompts from prompts.txt, prepend context from context.txt,
    run them through the model, and store responses in a JSON file.
    """
    # Read context and prompts
    context = read_file("context.txt")
    prompts = read_prompts("prompts.txt", numTempUnsafe=numTempUnsafe, numFrameUnsafe=numFrameUnsafe, numSafe=numSafe)

    if not prompts:
        print("No prompts found or error reading prompts file")
        return

    responses = []

    for i, promptTuple in enumerate(prompts):
        prompt, label = promptTuple
        try:
            # Reset model before each prompt
            print(f"\nResetting model context...")
            if not reset_model(model_name):
                print("Warning: Failed to reset model context")
                break

            # Combine context and prompt
            full_prompt = f"{context}\n\n{prompt}" if context else prompt

            data = {
                "model": model_name,
                "prompt": full_prompt
            }

            print(f"\nPrompt {i + 1}/{len(prompts)}:")
            print(f"Q: {prompt}")

            url = "http://localhost:11434/api/generate"
            response = requests.post(url, json=data)

            if response.status_code != 200:
                print(f"Error: Server returned status code {response.status_code}")
                print(f"Response: {response.text}")
                continue

            # Parse the response line by line and concatenate
            response_text = ""
            for line in response.text.strip().split('\n'):
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        response_text += json_response['response']
                except json.JSONDecodeError:
                    continue

            # Extract thinking and final response
            thinking = ""
            final_response = ""

            # Look for thinking section between <think> tags
            think_start = response_text.find("<think>")
            think_end = response_text.find("</think>")

            if think_start != -1 and think_end != -1:
                thinking = response_text[think_start + 7:think_end].strip()
                # Get the response after </think> tag
                final_response = response_text[think_end + 8:].strip().strip("ANS: ")
            else:
                # If no think tags found, store everything in final_response
                final_response = response_text.strip().strip("ANS: ")

            if final_response.startswith("o, "):  # Check for the broken "No"
                final_response = "N" + final_response  # Fix the broken "No"

            # Store prompt and response
            response_data = {
                "context": context,
                "prompt": prompt,
                "label": label,
                "full_prompt": full_prompt,
                "thinking_process": thinking,
                "final_response": final_response
            }
            responses.append(response_data)

            # Print both parts
            if thinking:
                print(f"Thinking process: {thinking}\n")

            # Fix the console print of final response
            console_response = final_response
            if console_response.startswith("o, "):
                console_response = "N" + console_response
            print(f"A: {console_response}\n")

            time.sleep(1)

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Make sure it's running on localhost:11434")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

    # Save all responses to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"\nResponses saved to {output_file}")
    except Exception as e:
        print(f"Error saving responses to file: {str(e)}")


if __name__ == '__main__':
    model_run(model_name="filter", numTempUnsafe=144, numFrameUnsafe=144, numSafe=144)
