# from diffusers import (
#     AutoPipelineForImage2Image,
#     AutoencoderTiny,
# )
# from compel import Compel
# import torch
#
# try:
#     import intel_extension_for_pytorch as ipex  # type: ignore
# except:
#     pass
#
# import psutil
# from config import Args
# from pydantic import BaseModel, Field
# from PIL import Image
#
# base_model = "SimianLuo/LCM_Dreamshaper_v7"
# taesd_model = "madebyollin/taesd"
#
# default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
#
#
# class Pipeline:
#     class Info(BaseModel):
#         name: str = "img2img"
#         title: str = "Image-to-Image LCM"
#         description: str = "Generates an image from a text prompt"
#         input_mode: str = "image"
#
#     class InputParams(BaseModel):
#         prompt: str = Field(
#             default_prompt,
#             title="Prompt",
#             field="textarea",
#             id="prompt",
#         )
#         seed: int = Field(
#             2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
#         )
#         steps: int = Field(
#             4, min=2, max=15, title="Steps", field="range", hide=True, id="steps"
#         )
#         width: int = Field(
#             512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
#         )
#         height: int = Field(
#             512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
#         )
#         guidance_scale: float = Field(
#             0.2,
#             min=0,
#             max=20,
#             step=0.001,
#             title="Guidance Scale",
#             field="range",
#             hide=True,
#             id="guidance_scale",
#         )
#         strength: float = Field(
#             0.5,
#             min=0.25,
#             max=1.0,
#             step=0.001,
#             title="Strength",
#             field="range",
#             hide=True,
#             id="strength",
#         )
#
#     def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
#         if args.safety_checker:
#             self.pipe = AutoPipelineForImage2Image.from_pretrained(base_model)
#         else:
#             self.pipe = AutoPipelineForImage2Image.from_pretrained(
#                 base_model,
#                 safety_checker=None,
#             )
#         if args.use_taesd:
#             self.pipe.vae = AutoencoderTiny.from_pretrained(
#                 taesd_model, torch_dtype=torch_dtype, use_safetensors=True
#             )
#
#         self.pipe.set_progress_bar_config(disable=True)
#         self.pipe.to(device="cuda", dtype=torch_dtype)
#         if device.type != "mps":
#             self.pipe.unet.to(memory_format=torch.channels_last)
#
#         # check if computer has less than 64GB of RAM using sys or os
#         if psutil.virtual_memory().total < 64 * 1024 ** 3:
#             self.pipe.enable_attention_slicing()
#
#         if args.torch_compile:
#             print("Running torch compile")
#             self.pipe.unet = torch.compile(
#                 self.pipe.unet, mode="reduce-overhead", fullgraph=True
#             )
#             self.pipe.vae = torch.compile(
#                 self.pipe.vae, mode="reduce-overhead", fullgraph=True
#             )
#
#             self.pipe(
#                 prompt="warmup",
#                 image=[Image.new("RGB", (768, 768))],
#             )
#
#         self.compel_proc = Compel(
#             tokenizer=self.pipe.tokenizer,
#             text_encoder=self.pipe.text_encoder,
#             truncate_long_prompts=False,
#         )
#
#     def predict(self, params: "Pipeline.InputParams") -> Image.Image:
#         generator = torch.manual_seed(params.seed)
#         prompt_embeds = self.compel_proc(params.prompt)
#         results = self.pipe(
#             image=params.image,
#             prompt_embeds=prompt_embeds,
#             generator=generator,
#             strength=params.strength,
#             num_inference_steps=params.steps,
#             guidance_scale=params.guidance_scale,
#             width=params.width,
#             height=params.height,
#             output_type="pil",
#         )
#
#         nsfw_content_detected = (
#             results.nsfw_content_detected[0]
#             if "nsfw_content_detected" in results
#             else False
#         )
#         if nsfw_content_detected:
#             return None
#
#         result_images = results.images
#
#         folder_path = "C://Users//Star//images"
#         # new start
#         # create folder if not exist
#         if not os.path.exists(folder_path):
#             os.mkdir(folder_path)
#
#         gif_images = []
#
#         for i, result_image in enumerate(result_images):
#             # save img - png
#             png_filename = f"result_image_{i}.png"
#             result_image.save(os.path.join(folder_path, png_filename))
#
#             # append
#             gif_images.append(result_image)
#
#         # save the img in gif
#         gif_filename = "result_images.gif"
#         gif_path = os.path.join(folder_path, gif_filename)
#
#         # convert to gif
#         gif_images[0].save(
#             gif_path,
#             save_all=True,
#             append_images=gif_images[1:],
#             duration=2000,
#             loop=0
#         )
#
#         return result_images
import os.path
from flask import Flask, jsonify

app = Flask(__name__)

from diffusers import (
    AutoPipelineForImage2Image,
    AutoencoderTiny,
)
from compel import Compel
import torch
from datetime import datetime

try:
    import intel_extension_for_pytorch as ipex  # type: ignore

except:
    pass

import psutil
import random
import string
from config import Args
from pydantic import BaseModel, Field
from PIL import Image

base_model = "SimianLuo/LCM_Dreamshaper_v7"
taesd_model = "madebyollin/taesd"

default_prompt = "Portrait of The Terminator with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"


class Pipeline:
    class Info(BaseModel):
        name: str = "img2img"
        title: str = "Image-to-Image LCM"
        description: str = "Generates an image from a text prompt"
        input_mode: str = "image"

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        seed: int = Field(
            2159232, min=0, title="Seed", field="seed", hide=True, id="seed"
        )
        steps: int = Field(
            4, min=2, max=15, title="Steps", field="range", hide=True, id="steps"
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        guidance_scale: float = Field(
            0.2,
            min=0,
            max=20,
            step=0.001,
            title="Guidance Scale",
            field="range",
            hide=True,
            id="guidance_scale",
        )
        strength: float = Field(
            0.5,
            min=0.25,
            max=1.0,
            step=0.001,
            title="Strength",
            field="range",
            hide=True,
            id="strength",
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        # image_db = [{"user_id": "", "image": ""}]
        self.image_db = []
        self.output_gif = []

        if args.safety_checker:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(base_model)
        else:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                base_model,
                safety_checker=None,
            )
        if args.use_taesd:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_model, torch_dtype=torch_dtype, use_safetensors=True
            )

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=device, dtype=torch_dtype)
        if device.type != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        # check if computer has less than 64GB of RAM using sys or os
        if psutil.virtual_memory().total < 64 * 1024 ** 3:
            self.pipe.enable_attention_slicing()

        if args.torch_compile:
            print("Running torch compile")
            self.pipe.unet = torch.compile(
                self.pipe.unet, mode="reduce-overhead", fullgraph=True
            )
            self.pipe.vae = torch.compile(
                self.pipe.vae, mode="reduce-overhead", fullgraph=True
            )

            self.pipe(
                prompt="warmup",
                image=[Image.new("RGB", (768, 768))],
            )

        self.compel_proc = Compel(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            truncate_long_prompts=False,
        )

    def count_user_id_occurrences(self, user_id):
        count = sum(1 for image_data in self.image_db if image_data.get(
            "user_id") == user_id)
        return count

    def add_image(self, user_id, result_image):
        self.image_db.append({"user_id": user_id, "image": result_image})

    def get_user_image(self, user_id):
        # Filter images by the given user_id and retrieve the last three images
        filtered_images = [image_data['image'] for image_data in reversed(
            self.image_db) if image_data.get('user_id') == user_id][:3]
        return filtered_images

    def generate_gif(self, user_id, duration=500,
                     output_folder="public/download"):
        image_list = self.get_user_image(user_id)

        if not image_list:
            print("No images found for the user.")
            return

        # Create a list to hold image objects
        frames = []
        frames = image_list

        if not frames:
            print("No valid images found for the user.")
            return

        # Save the frames as a GIF
        random_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

        # Combine the random string with the GIF extension
        output_filename = f"{random_string}.gif"
        # output_filename = f"{datetime.utcnow()}.gif"

        frames[0].save(os.path.join(output_folder, output_filename), save_all=True,
                       append_images=frames[1:], duration=duration, loop=0)
        print(f"Generated GIF: {os.path.join(output_folder, output_filename)}")

        # return os.path.join(output_folder, output_filename)
        return output_filename

    def get_dynamic_filename(self):
        if len(self.output_gif) > 0:
            dynamic_filename = self.output_gif[0].get("output", None)
            if dynamic_filename:
                return {"dynamic_filename": dynamic_filename}

    def predict(self, params: "Pipeline.InputParams", user_id) -> Image.Image:
        generator = torch.manual_seed(params.seed)
        prompt_embeds = self.compel_proc(params.prompt)
        results = self.pipe(
            image=params.image,
            prompt_embeds=prompt_embeds,
            generator=generator,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance_scale,
            width=params.width,
            height=params.height,
            output_type="pil",
        )

        nsfw_content_detected = (
            results.nsfw_content_detected[0]
            if "nsfw_content_detected" in results
            else False
        )
        if nsfw_content_detected:
            return None
        result_image = results.images[0]

        self.add_image(user_id, result_image)

        user_snap_count = self.count_user_id_occurrences(user_id)
        if (user_snap_count > 2 and user_snap_count < 4):
            output_filename = self.generate_gif(user_id)
            self.output_gif.append({"user_id": user_id, "output": output_filename})

        return result_image
