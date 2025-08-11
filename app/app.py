import gradio as gr
import numpy as np
from PIL import Image

# Placeholder demo so it always runs.
# Replace with: from src.inference import run_inference
# and return run_inference(seed)
def generate(seed: int = 42):
    rng = np.random.default_rng(int(seed))
    img = (rng.random((256, 256, 3)) * 255).astype("uint8")
    return Image.fromarray(img)

demo = gr.Interface(
    fn=generate,
    inputs=gr.Number(value=42, label="Seed"),
    outputs=gr.Image(label="Output"),
    title="Face Synthesis — Diffusion/GAN (Demo)",
    description="Starter demo. I’ll wire in my real diffusion sampler next."
)

if __name__ == "__main__":
    demo.launch()