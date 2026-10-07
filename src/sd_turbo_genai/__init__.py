import torch 
from diffusers import AutoPipelineForText2Image

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32

)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear")
negative_prompt = ("blurry, low quality, worst quality, distorted face, deformed anatomy, bad proportions, extra fingers, missing fingers, fused fingers, bad hands, long neck, plastic skin, waxy skin, watermark, text, logo, signature")

print("Generando la imagen ... :)")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,
    guidance_scale=2.0,
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")

print("Imagen guardada.")
