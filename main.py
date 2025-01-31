import requests
import json
from typing import List, Dict
import time


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


def read_prompts(file_path: str = "prompts.txt") -> List[str]:
    """
    Read prompts from a text file, one prompt per line.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        return prompts
    except Exception as e:
        print(f"Error reading prompts file: {str(e)}")
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


def model_run(output_file: str = "responses.json", model_name: str = "llama2-uncensored") -> None:
    """
    Read prompts from prompts.txt, prepend context from context.txt,
    run them through the model, and store responses in a JSON file.
    """
    # Read context and prompts
    context = read_file("context.txt")
    prompts = read_prompts()

    if not prompts:
        print("No prompts found or error reading prompts file")
        return

    responses = []

    for i, prompt in enumerate(prompts):
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

            # Use the complete URL
            url = "http://localhost:11434/api/generate"
            response = requests.post(url, json=data)

            if response.status_code != 200:
                print(f"Error: Server returned status code {response.status_code}")
                print(f"Response: {response.text}")
                continue

            # Parse the response line by line as it might contain multiple JSON objects
            response_text = ""
            for line in response.text.strip().split('\n'):
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        response_text += json_response['response']
                except json.JSONDecodeError:
                    continue

            # Store prompt and response
            response_data = {
                "context": context,
                "prompt": prompt,
                "full_prompt": full_prompt,
                "response": response_text
            }
            responses.append(response_data)

            print(f"A: {response_text}\n")

            # Add a small delay between requests
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
    model_run(model_name="filter")