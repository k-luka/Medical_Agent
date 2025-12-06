import os
from pathlib import Path
from google import genai
from google.genai import types
import hydra
import warnings
from omegaconf import DictConfig
# tools
from src.tools.inference import load_models_from_hydra
from src.tools.defenitions import (
    segment_spleen_ct,
    segment_multi_organ_ct,
    lung_nodule_ct_detection,
    inspect_segmentation_tool,
    view_saved_image,
    init_paths,
)

warnings.filterwarnings("ignore", module="monai")
import logging

logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"--- Starting Medical Agent (UI: {cfg.ui.type}) ---")

    # load models
    load_models_from_hydra(cfg)
    init_paths(Path(cfg.sandbox.path))

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # create the client
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # define tools
    my_tools = [
        segment_spleen_ct, 
        # segment_brain_mri,
        segment_multi_organ_ct,
        lung_nodule_ct_detection,
        inspect_segmentation_tool,
        view_saved_image,
    ]

    generate_config = types.GenerateContentConfig(
        temperature=cfg.llm.temperature,
        tools=my_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
        system_instruction = cfg.llm.system_prompt
    )

    chat = client.chats.create(model=cfg.llm.model_id, config=generate_config)

    print("---------------------------------------")
    print("+++ Agent Ready. Models are loaded. +++")

    while True:
        try:
            user_input = input("\n> User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting Medical Agent. Goodbye!")
                break
            
            response = chat.send_message(user_input)
            print(f"\n> Agent: {response.text}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
