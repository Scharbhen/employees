from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import pandas as pd
import json
from datetime import datetime
import time

def predict(image_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=5000)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

images_directory = 'Images/'
results_directory = 'Results/'
question_path = 'question.txt'

df = pd.DataFrame(columns=['Номер акта', 'Дата акта', 'Наименование товара или услуги', 'Количество',
                           'Единица измерения', 'Цена за единицу', 'Стоимость проданных товаров или услуг'])

start_time = time.time()
for file in os.listdir(images_directory):
    with open(question_path, 'r') as question_file:
        question = question_file.read()
    answer = predict(images_directory + file, question)
    answer_json = answer[0][answer[0].find('{'):answer[0].rfind('}')+1]
    data = json.loads(answer_json, strict=False)
    for product in data["Товары"]:
        df = pd.concat([df, pd.DataFrame([{'Номер акта' : data["Номер акта"], 'Дата акта' : data["Дата акта"],
                                           'Наименование товара или услуги' : product["Наименование"],
                                           'Количество' : product["Количество"],
                                           'Единица измерения' : product["Единица измерения"],
                                           'Цена за единицу' : product["Цена за единицу"],
                                           'Стоимость проданных товаров или услуг' : product["Сумма"]}])],
                       ignore_index=True)
df.to_excel(results_directory +'result_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.xlsx', index=False)
print("Время обработки в секундах: ", time.time() - start_time)