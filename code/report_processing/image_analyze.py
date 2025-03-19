import os
from dashscope import MultiModalConversation
import dashscope
import fitz  
from PIL import Image
import io

dashscope.api_key = ""
def convert_image_to_jpg(img_bytes, output_path):
    """
    将图像转换为JPG格式，并保存。
    :param img_bytes: 图像的字节数据
    :param output_path: 输出路径
    :return: None
    """
    try:
        image = Image.open(io.BytesIO(img_bytes))
        image = image.convert('RGB')
        image.save(output_path, 'JPEG')
        print(f"Converted image saved as {output_path}")
        return output_path  # 返回保存的路径
    except Exception as e:
        print(f"Error converting image: {e}")
        return None  
def call_with_local_file(image_paths):
    """
    使用多模态大模型对传入的图片路径进行分析。
    :param image_paths: 包含图片路径的列表
    :return: 分析结果
    """
    if not image_paths:
        print("No image paths provided for analysis!")
        return None

    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }]

    user_message = {
        'role': 'user',
        'content': [{'image': img_path} for img_path in image_paths] + [
            {'text': '请你详细分析图片的内容?'}
        ]
    }
    messages.append(user_message)
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    print("API response:", response)
    return response
def extract_images_from_pdf_and_analyze(pdf_path, output_folder):
    """
    从PDF文件中提取图像，并使用多模态大模型进行分析。
    :param pdf_path: PDF文件路径
    :param output_folder: 保存图片的文件夹
    :return: 返回图像分析结果
    """
    doc = fitz.open(pdf_path)
    image_paths = []
    image_page_mapping = []  # 用于存储图片和其对应的PDF页码

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            img_base = doc.extract_image(img[0])
            img_bytes = img_base["image"]
            img_ext = img_base["ext"]
            width = img_base["width"]
            height = img_base["height"]
            img_size = len(img_bytes)

            if width == 0 or height == 0 or img_size < 1024:  # 小于 1KB 的图片视为无效
                print(f"Skipped empty or very small image on page {page_num + 1}")
                continue

            img_name = f"page{page_num + 1}_img{img_index + 1}.jpg"  
            img_path = os.path.join(output_folder, img_name)

            converted_image_path = convert_image_to_jpg(img_bytes, img_path)

            if not converted_image_path:
                continue

            image_paths.append(f'file://{os.path.abspath(converted_image_path)}')
            image_page_mapping.append(page_num + 1) 
    doc.close()

    # 调用多模态大模型进行分析
    analysis_data = []
    if image_paths:
        for idx, img_path in enumerate(image_paths):
            try:
                analysis_results = call_with_local_file([img_path])

                if analysis_results and 'output' in analysis_results and 'choices' in analysis_results['output']:
                    analysis_text = analysis_results['output']['choices'][0]['message']['content'][0]['text']
                    page_num = image_page_mapping[idx] 
                    analysis_data.append({
                        'document': pdf_path,
                        'page': page_num, 
                        'image_index': idx + 1,
                        'analysis': analysis_text
                    })
                else:
                    print(f"No valid analysis for image on page {image_page_mapping[idx]}")
                    page_num = image_page_mapping[idx]
                    analysis_data.append({
                        'document': pdf_path,
                        'page': page_num,
                        'image_index': idx + 1,
                        'analysis': 'No valid analysis available.'
                    })

            except Exception as e:
                print(f"Error analyzing image {idx + 1} on page {image_page_mapping[idx]}: {e}")
                continue 

    if not analysis_data:
        print("No images were successfully analyzed.")
    return analysis_data

