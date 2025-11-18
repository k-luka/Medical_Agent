from src.gemini_client import run_gemini

reply = run_gemini("Explain what segmentation models do in one sentence.")

print("Gemini Response:", reply)