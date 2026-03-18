import os


def get_hf_token(required=False):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if required and not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
        )
    return token
