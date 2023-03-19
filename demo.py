import openai
import requests
openai.api_key =

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1, # 生成的图像数量
        size="512x512", # 图像大小
    )

    # 从 API 响应中获取图像 URL
    image_url = response['data'][0]['url']

    # 从 URL 中下载图像并保存到本地
    image_data = requests.get(image_url).content
    with open('generated_image.jpg', 'wb') as f:
        f.write(image_data)

generate_image("a red apple on a white background") # 生成一张描述为 "a red apple on a white background" 的图像并保存到本地
