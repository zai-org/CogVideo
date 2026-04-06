"""
THis is the main file for the gradio web demo. It uses the CogVideoX-2B model to generate videos gradio web demo.

The optional prompt enhancement feature uses an LLM to refine your prompt before video generation.
Set one of the following environment variables to enable it:

  - MINIMAX_API_KEY: Use MiniMax (MiniMax-M2.7) as the LLM provider.
      MINIMAX_API_KEY=your_minimax_api_key python inference/gradio_web_demo.py

  - OPENAI_API_KEY: Use OpenAI or any OpenAI-compatible provider.
      OPENAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py

This demo only supports the text-to-video generation model.
If you wish to use the image-to-video or video-to-video generation models,
please use the gradio_composite_demo to implement the full GUI functionality.
"""

import os
import threading
import time

import gradio as gr
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from datetime import datetime, timedelta
from openai import OpenAI
from moviepy import VideoFileClip

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to(
    "cuda"
)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def _get_llm_client():
    """
    Return (client, model_name) based on environment variables.

    Priority:
      1. MINIMAX_API_KEY → MiniMax (OpenAI-compatible, https://api.minimax.io/v1)
      2. OPENAI_API_KEY  → OpenAI or any OpenAI-compatible provider via OPENAI_BASE_URL
    """
    minimax_api_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_api_key:
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        client = OpenAI(api_key=minimax_api_key, base_url=base_url)
        return client, "MiniMax-M2.7"
    return OpenAI(), "glm-4-plus"


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("MINIMAX_API_KEY"):
        return prompt

    client, model = _get_llm_client()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model=model,
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    torch.cuda.empty_cache()
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
    ).frames[0]

    return video


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path


def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    clip = clip.with_fps(8)
    clip = clip.resized(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX Gradio Simple Space🤗
            """)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5
            )

            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            with gr.Column():
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                    "50 steps are recommended for most cases.<br>"
                )
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    def generate(
        prompt,
        num_inference_steps,
        guidance_scale,
        model_choice,
        progress=gr.Progress(track_tqdm=True),
    ):
        tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)

        return video_path, video_update, gif_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=[video_output, download_video_button, download_gif_button],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])

if __name__ == "__main__":
    demo.launch()
