import argparse
import google.generativeai as genai
import os

def main():
    parser = argparse.ArgumentParser(description="List available Google Generative AI models for a given API key")
    parser.add_argument("--api-key", help="Google API key (or use GOOGLE_API_KEY environment variable)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Please provide a valid API key via --api-key or set the GOOGLE_API_KEY environment variable.")
        return

    genai.configure(api_key=api_key)

    models = genai.list_models()

    print("=== Available Models for your API Key ===\n")
    for m in models:
        print(m.name)

if __name__ == "__main__":
    main()
