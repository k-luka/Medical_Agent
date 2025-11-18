import os
from google import genai
from google.genai import types
from tools.test_func_call import segment_image

# def get_client():
#     api_key = os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("API key not found in environment variables.")

#     return genai.Client(api_key=api_key)

# def run_gemini(prompt: str, model_id="gemini-2.5-flash", temperature: float = 0.0):
#     client = get_client()

#     response = client.models.generate_content(
#         model=model_id,
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=temperature
#         )
#     )
#     return response.text

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Create the config
    # We pass the actual python fucntion into 'tools'.
    # We set 'automatic_function_calling' to NOT disabled
    config = types.GenerateContentConfig(
        tools = [segment_image],
        automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=False)
    )

    # Make one call
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = "Segment the liver in /tmp/example_ct.nii.gz and explain the result.",
        config = config,
    )

    print("Agent Response:", response.text)

if __name__ == "__main__":
    main()


