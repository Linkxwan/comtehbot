from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = [
    "http://localhost",  # Для локальной разработки
    "http://localhost:8000",  # Для локальной разработки
    "*",  # Разрешить все домены (можно уточнить для безопасности)
]

# Добавление CORS middleware в приложение
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Список разрешенных источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)



def extract_colors(image: Image.Image, n_clusters: int = 10):
    # Преобразуем изображение в массив
    data = np.array(image)
    pixels = data.reshape(-1, 3)
    
    # Используем K-Means для извлечения основных цветов
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    
    # Получаем цвета
    colors = kmeans.cluster_centers_.astype(int)
    return colors.tolist()

def create_gradient(color_start, color_end, num_steps):
    color_start = np.array(color_start)  # Convert to NumPy array
    color_end = np.array(color_end)  # Convert to NumPy array
    return [list(color_start + (color_end - color_start) * i / (num_steps - 1)) for i in range(num_steps)]


@app.post("/extract_colors/")
async def get_image_colors(file: UploadFile = File(...), n_clusters: int = 10):
    # Читаем изображение из файла
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))
    
    # Извлекаем основные цвета
    colors = extract_colors(image, n_clusters)

    # Создаем 4 промежуточных значения между color1 и color2, и между color2 и color3
    gradient1_akcent = create_gradient(colors[0], colors[1], 6)
    gradient2_akcent = create_gradient(colors[1], colors[2], 6)[::-1]  # Убираем первый цвет (он дублирует color2)

    # Объединяем все цвета в один список
    colors_akcent = np.array(gradient1_akcent + gradient2_akcent).astype(int)

    # Создаем 4 промежуточных значения между color1 и color2, и между color2 и color3
    dark = create_gradient(np.array([0, 0, 0]), colors[0], 12)
    light = create_gradient(np.array([255, 255, 255]), colors[0], 12)

    # Вычисляем противоположные цвета
    opposite_colors = [255 - np.array(color) for color in colors]
    
    # Преобразуем NumPy массивы в списки для JSON сериализации
    return JSONResponse(content={
        "colors": colors,  # No need to convert colors, it's already a list
        "colors_akcent": colors_akcent.tolist(),  # Convert NumPy array to list
        "dark": dark,  # dark is already a list, no conversion needed
        "light": light,  # light is already a list, no conversion needed
        "opposite_colors": [color.tolist() for color in opposite_colors]  # Convert each opposite color to list
    })

    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    # pip install fastapi uvicorn pillow numpy scikit-learn