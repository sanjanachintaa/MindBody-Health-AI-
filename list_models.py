#!/usr/bin/env python3
"""List available Gemini models for your API key."""

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env or environment")
    exit(1)

import google.generativeai as genai
genai.configure(api_key=api_key)

print("Available models:\n")
for model in genai.list_models():
    print(f"  - {model.name}")
    print(f"    Display name: {model.display_name}")
    print(f"    Supports generateContent: {model.supported_generation_methods}")
    print()
