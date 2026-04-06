"""
The CogVideoX model is designed to generate high-quality videos based on detailed and highly descriptive prompts.
The model performs best when provided with refined, granular prompts, which enhance the quality of video generation.
This script is designed to assist with transforming simple user inputs into detailed prompts suitable for CogVideoX.
It can handle both text-to-video (t2v) and image-to-video (i2v) conversions.

- For text-to-video, simply provide the prompt.
- For image-to-video, provide the path to the image file and an optional user input.
The image will be encoded and sent as part of the request to Azure OpenAI.

### How to run:
Run the script for **text-to-video**:
    $ python convert_demo.py --prompt "A girl riding a bike." --type "t2v"

Run the script for **image-to-video**:
    $ python convert_demo.py --prompt "the cat is running" --type "i2v" --image_path "/path/to/your/image.jpg"

### Using MiniMax as the LLM provider:
    $ MINIMAX_API_KEY=your_minimax_api_key python convert_demo.py --prompt "A girl riding a bike." --type "t2v"

### Using OpenAI or any OpenAI-compatible provider:
    $ OPENAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 python convert_demo.py --prompt "A girl riding a bike." --type "t2v"
"""

import argparse
import os
from openai import OpenAI, AzureOpenAI
import base64
from mimetypes import guess_type


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

sys_prompt_t2v = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

sys_prompt_i2v = """
**Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input.

**Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty.

**Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

**Answering Style**:
Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

**Output Format**: "[highly descriptive image caption here]"

user input:
"""


def image_to_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def convert_prompt(prompt: str, retry_times: int = 3, type: str = "t2v", image_path: str = None):
    """
    Convert a prompt to a format that can be used by the model for inference.

    LLM provider is selected automatically from environment variables:
      - MINIMAX_API_KEY → MiniMax (MiniMax-M2.7)
      - OPENAI_API_KEY  → OpenAI or any OpenAI-compatible provider
    """
    client, default_model = _get_llm_client()
    ## To use Azure OpenAI instead, replace the line above with:
    # client = AzureOpenAI(api_key="", api_version="", azure_endpoint="")
    # default_model = "gpt-4o"

    text = prompt.strip()
    for i in range(retry_times):
        if type == "t2v":
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"{sys_prompt_t2v}"},
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
                    },
                    {
                        "role": "assistant",
                        "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
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
                        "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {text} "',
                    },
                ],
                model=default_model,
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
            )
        else:
            response = client.chat.completions.create(
                model=default_model,
                messages=[
                    {"role": "system", "content": f"{sys_prompt_i2v}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_url(image_path),
                                },
                            },
                        ],
                    },
                ],
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
            )
        if response.choices:
            return response.choices[0].message.content
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to convert")
    parser.add_argument(
        "--retry_times", type=int, default=3, help="Number of times to retry the conversion"
    )
    parser.add_argument("--type", type=str, default="t2v", help="Type of conversion (t2v or i2v)")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image file")
    args = parser.parse_args()

    converted_prompt = convert_prompt(args.prompt, args.retry_times, args.type, args.image_path)
    print(converted_prompt)
