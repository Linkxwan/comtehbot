import json
from pydantic import BaseModel

import logging
import pickle
import time
import uuid
from typing import Dict, List, Tuple, Union

import aiofiles
import edge_tts
import g4f
import httpx
import scipy as sp
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request, Form, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

from data._data_contexts import contexts


# Настройка логгера
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация переменных и настроек
MAX_QUESTION_HISTORY_SIZE = 10  # Максимальный размер списка

# Создаем пустой список для хранения значений переменной question
question_history = []


synthesis_path="voice"
# Проверяем, существует ли папка
if not os.path.exists(synthesis_path):
    # Создаем новую папку
    os.makedirs(synthesis_path)
    print(f"Папка {synthesis_path} успешно создана")
else:
    print(f"Папка {synthesis_path} уже существует")


# Инициализация FastAPI приложения
app = FastAPI()


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
# Предоставляем папку "voice" в пользование как статические файлы
app.mount("/voice", StaticFiles(directory="voice"), name="voice")

# Словарь для хранения истории чата каждого пользователя
chat_history_by_user: Dict[str, List[str]] = {}


# Логируем успешную инициализацию переменных и настроек
logging.info("Успешно инициализированы переменные и настройки.")


# ===== ЗАГРУЗКА МОДЕЛИ НЕЙРОННОЙ СЕТИ =====
# vocab_size = 15175
max_seq_length = 75
# Загрузка токенизатора из файла pickle
with open('data\\tokenizer_003.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Путь к вашей папке saved_model
saved_model_path = 'data\\comtehbot_haha.keras'
# Загрузка модели
model = tf.keras.models.load_model(saved_model_path)


# ===== ЗАГРУЗКА БАЗЫ ДАННЫХ =====
# Загрузка данных из первого файла
with open('data\\combined_file.json', 'r', encoding='utf-8') as file:
    data_combined = json.load(file)


# Получение списка вопросов
questions = [item['question'] for item in data_combined]


# ===== ВЕКТОРИЗАЦИЯ ЗНАЧЕНИЙ ИЗ БД =====
# Создание объекта TfidfVectorizer для векторизации ключей из data_set
tfidf_vectorizer = TfidfVectorizer()
# Преобразование данных в векторную форму
try:
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    logging.info("Успешно выполнено векторизация базы_1!>")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации базы_1!>: {e}")

# Создание объекта TfidfVectorizer для векторизации текстов из contexts
contexts_vectorizer = TfidfVectorizer()
# Преобразование текстов из contexts в векторную форму
try:
    contexts_vectors = contexts_vectorizer.fit_transform(contexts)
    logging.info("Успешно выполнено векторизация текстов!>")
except Exception as e:
    logging.error(f"Ошибка при выполнении векторизации текстов!>: {e}")


def load_data(file_name: str) -> dict:
    """
    Загружает данные из файла JSON.

    Args:
        file_name (str): Путь к файлу JSON.

    Returns:
        dict: Загруженные данные или пустой словарь, если файл не найден.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    return data

def save_data(data: dict, file_name: str) -> None:
    """
    Сохраняет данные в файл JSON.

    Args:
        data (dict): Данные для сохранения.
        file_name (str): Путь к файлу JSON.
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def add_entry(dictionary: dict, key: str, value) -> None:
    """
    Добавляет запись в словарь.

    Если ключ уже существует, значение добавляется в список значений этого ключа.

    Args:
        dictionary (dict): Словарь, в который нужно добавить запись.
        key (str): Ключ, по которому нужно добавить значение.
        value: Значение, которое нужно добавить в словарь.
    """
    if key in dictionary:
        if value not in dictionary[key]:
            dictionary[key].append(value)
    else:
        dictionary[key] = [value]

# Путь к файлу данных
file_name = "data.json"

# Загрузка данных из файла
data = load_data(file_name)


async def process_chunk(voice: str, text_chunk: str, output_file: str) -> None:
    """
    Обрабатывает фрагмент текста, используя указанный голос, и сохраняет результат в файле.

    Args:
        voice (str): Голос для синтеза речи.
        text_chunk (str): Фрагмент текста для обработки.
        output_file (str): Путь к файлу для сохранения результата.
    """
    communicate = edge_tts.Communicate(text_chunk, voice)
    await communicate.save(output_file)


async def synthesis(data: str, prefix: str = synthesis_path) -> Tuple[List[str], str]:
    """
    Синтезирует речь из текста, используя асинхронный метод.

    Args:
        data (str): Текст для синтеза речи.
        prefix (str, optional): Префикс пути к сохраненным файлам. По умолчанию synthesis_path.

    Returns:
        Tuple[List[str], str]: Список созданных файлов и уникальный идентификатор.
    """
    voice = 'ru-RU-SvetlanaNeural'  # Установите желаемый голос
    unique_id = uuid.uuid4()
    created_files = []  # Список для хранения созданных файлов

    # Создаем и запускаем асинхронные потоки для каждого фрагмента текста
    output_file = os.path.join(prefix, f"synthesis_{unique_id}.mp3")  # Уникальный путь к файлу
    await process_chunk(voice, data, output_file)

    # Собираем имена созданных файлов
    output_file = os.path.join(prefix, f"synthesis_{unique_id}.mp3")
    created_files.append(output_file)

    return created_files, unique_id


async def vectorize(question: str, 
                    tfidf_vectorizer: TfidfVectorizer, 
                    tfidf_matrix: sp.sparse.csr_matrix) -> Tuple[int, float]:
    """
    Векторизует вопрос и сравнивает его с данными из базы.

    Args:
        question (str): Вопрос, который необходимо векторизовать и сравнить.
        tfidf_vectorizer (TfidfVectorizer): Объект TfidfVectorizer для векторизации текста.
        tfidf_matrix (sparse matrix): Матрица TF-IDF, представляющая данные из базы.

    Returns:
        Tuple[int, float]: Индекс наиболее похожего вопроса из базы и процент сходства.
    """
    # Векторизация вопроса
    question_vector = tfidf_vectorizer.transform([question])

    # Вычисление косинусного сходства между вопросом и данными из базы
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Находим индекс наиболее похожего вопроса
    most_similar_index = cosine_similarities.argmax()
    
    # Возвращаем индекс и процент сходства
    return most_similar_index, cosine_similarities[most_similar_index]


async def remove_files(files: List[str], prefix: str = synthesis_path) -> None:
    """
    Асинхронно удаляет файлы из указанной папки с префиксом.

    Args:
        files (List[str]): Список имен файлов, которые нужно удалить.
        prefix (str, optional): Префикс пути к файлам. По умолчанию synthesis_path.
    """
    try:
        # Удаление каждого файла
        for file in files:
            file_path = os.path.join(prefix, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Файл {file_path} удален успешно.")
            else:
                logging.warning(f"Файл {file_path} не существует.")
    except Exception as e:
        # Логирование ошибки, если что-то пошло не так
        logging.error(f"Ошибка при удалении файлов: {e}")


async def append_question_history_to_file() -> None:
    """
    Дополняет файл новыми значениями переменной question.

    Эта функция асинхронно записывает новые вопросы из переменной question_history
    в файл "question_history.txt".

    """
    async with aiofiles.open("question_history.txt", "a", encoding="utf-8") as file:
        for question in question_history:
            await file.write(question + "\n")



@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Обрабатывает событие остановки сервера.

    Эта функция вызывается при остановке сервера и асинхронно сохраняет
    историю вопросов в файл перед завершением работы сервера.

    """
    await append_question_history_to_file()
    logging.info("Сервер остановлен.")


@app.get("/dtxt")
async def download_txt(response: Response):
    """
    Скачивает текстовый файл.

    Args:
        response (Response): Объект ответа FastAPI.

    Returns:
        FileResponse: Ответ с текстовым файлом для скачивания.
    """
    file_path = "question_history.txt"  # Укажите путь к файлу
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="question_history.txt", media_type='application/octet-stream')
    else:
        response.status_code = 404
        return {"error": "File not found"}
    
@app.get("/dbad")
async def download_bad_answers_txt(response: Response):
    """
    Скачивает текстовый файл.

    Args:
        response (Response): Объект ответа FastAPI.

    Returns:
        FileResponse: Ответ с текстовым файлом для скачивания.
    """
    file_path = "bad_answers.txt"  # Укажите путь к файлу
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="bad_answers.txt", media_type='application/octet-stream')
    else:
        response.status_code = 404
        return {"error": "File not found"}

@app.get("/djson")
async def download_json(response: Response):
    """
    Скачивает файл JSON.

    Args:
        response (Response): Объект ответа FastAPI.

    Returns:
        FileResponse: Ответ с файлом JSON для скачивания.
    """
    file_path = "data.json"  # Укажите путь к файлу
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename="data.json", media_type='application/octet-stream')
    else:
        response.status_code = 404
        return {"error": "File not found"}
    
presentation_block = {
    'Какие специальности есть в колледже?': 'Вся необходимая информация на ссылке: https://comtehno.kg/specialties/',
    'Сколько стоит в этом учебном году контракт?': 'Оплата за обучение составляет 40 тысяч сом в год.',
    'Приемная комиссия': 'Whatsapp 0707379957',
    'Какие документы подать при поступлении?': 'Вся необходимая информация на ссылке: https://comtehno.kg/selection-committee/'
}


@app.get("/get_response")
async def get_response(question: str) -> Dict[str, Union[str, int, float]]:
    """
    Получает ответ на заданный вопрос.

    Args:
        question (str): Вопрос, на который требуется получить ответ.

    Returns:
        Dict[str, Union[str, int, float]]: Словарь с данными ответа и метаданными.
            Возможные ключи:
                - "question": Вопрос, на который получен ответ.
                - "response": Ответ на вопрос.
                - "accuracy_percentage" (опционально): Процент сходства для ответа, если применимо.
                - "time": Время, затраченное на обработку запроса в секундах.
    """
    try:
        start = time.time()
        key = question.split()[0]

        # Добавляем значение question в историю
        question_history.append(question)

        # Если размер списка достиг максимального значения, записываем его в файл и очищаем
        if len(question_history) >= MAX_QUESTION_HISTORY_SIZE:
            await append_question_history_to_file()
            question_history.clear()

        print(len(question_history))
        if key in ['/gpt', '/гпт']:
            # Логика для обработки запроса с использованием GPT
            question = question[4:].strip()
            context = contexts[cosine_similarity(contexts_vectors, contexts_vectorizer.transform([question])).argmax()]
            template = f"""Ты - полезный ИИ ассистент для нашего колледжа комтехно (comtehno). Тебя зовут виртуальный ассистент комтехно. Ты женского пола, отвечай как девушка.
                Используй следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
                Если тебе задают вопросы связанные с программированием и колледжем отвечай. Если не знаешь что ответить отправь их на сайт колледжа https://www.comtehno.kg.
                Context: {context}
                Question: {question}
                Answer:"""

            # Генерация речевого ответа
            response = await g4f.ChatCompletion.create_async(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": template}]
            )

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": response,
                "time": time.time() - start,
            }
        
        elif key in ['/lstm']:
            # Логика для обработки запроса с использованием LSTM
            question = question[5:].strip()
            question_sequence = tokenizer.texts_to_sequences([question])
            question_sequence = tf.keras.utils.pad_sequences(question_sequence, maxlen=max_seq_length, padding='post')
            # Преобразование входных данных в тензор TensorFlow
            question_tensor = tf.convert_to_tensor(question_sequence, dtype=tf.float32)
            predicted_answer_sequence = model.predict(question_tensor)
            predicted_word_indices = predicted_answer_sequence.argmax(axis=-1)
            predicted_words = [tokenizer.index_word[idx] for idx in predicted_word_indices.flatten() if idx != 0]  # Преобразуем индексы в слова
            predicted_answer = ' '.join(predicted_words)  # Объединяем слова в предложение

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": predicted_answer,
                "time": time.time() - start
            }
        else:
            # Логика для обработки запроса с использованием TF-IDF и нахождения наиболее подходящего ответа
            most_similar_index, accuracy = await vectorize(question, tfidf_vectorizer, tfidf_matrix)
            most_similar_answer = data_combined[most_similar_index]['answer']
            if most_similar_answer == 'Абу-Даби':
                most_similar_answer = 'Простите, я не совсем поняла, что вы имели в виду. Можете объяснить подробнее? Спасибо!'

            # Возвращаем ответ с данными и метаданными
            return {
                "question": question,
                "response": most_similar_answer,
                "accuracy_percentage": "{:.2f}%".format(accuracy * 100),
                "time": time.time() - start,
            }
        
    except Exception as e:
        # Логируем ошибку
        logging.error(f"Произошла ошибка при обработке вопроса: {question}. Ошибка: {e}")
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, response: Response):
    """
    Возвращает HTML-страницу для домашней страницы.

    Args:
        request (Request): Объект запроса FastAPI.
        response (Response): Объект ответа FastAPI.

    Returns:
        TemplateResponse: HTML-страница для домашней страницы.
    """
    user_id = request.cookies.get("user_id") or str(uuid.uuid4())
    chat_history = chat_history_by_user.setdefault(user_id, [])
    message_count = len(chat_history)
    response = templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history, "message_count": message_count})
    response.set_cookie("user_id", user_id)
    return response


@app.post("/api", response_class=HTMLResponse)
async def get_answer(request: Request, question: str = Form(...)) -> RedirectResponse:
    """
    Получает ответ на заданный вопрос и обновляет чат-историю.

    Args:
        request (Request): Объект запроса FastAPI.
        question (str): Вопрос, на который требуется получить ответ.

    Returns:
        RedirectResponse: Перенаправление на домашнюю страницу.
    """
    user_id = request.cookies.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=400, detail="Session ID is missing")
    if not question:
        return RedirectResponse("/", status_code=303)

    chat_history = chat_history_by_user.setdefault(user_id, [])
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://127.0.0.1:8000/get_response", params={"question": question}, timeout=20)
            response.raise_for_status()
            response_data = response.json()
            if "response" in response_data:
                chat_history.append({"role": "user", "message": question})
                chat_history.append({"role": "assistant", "message": response_data["response"]})
                chat_history_by_user[user_id] = chat_history

                # Добавляем запись в словарь.
                add_entry(data, user_id, chat_history)
                # Сохранение данных в файл
                save_data(data, file_name)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")

    return RedirectResponse("/", status_code=303)


@app.get("/synthesis")
async def process_request(data: str):
    """
    Обрабатывает запрос на синтез речи на основе переданных данных.

    Args:
        data (str): Текст, для которого требуется синтез речи.

    Returns:
        Union[Dict[str, str], Dict[str, str]]: Словарь с результатом запроса или сообщением об ошибке.
    """
    try:
        created_file = await synthesis(data)
        # Пример возврата успешного ответа
        return {"created_file": created_file[0]}
    
    except Exception as e:
        return {"error": str(e)}


class AnswerData(BaseModel):
    """
    Модель данных для обработки запросов в функции bad_answer_process.

    Args:
    - user (str): Текст сообщения пользователя.
    - assistant (str): Текст ответа ассистента.
    """
    user: str
    assistant: str


@app.post("/bad_answer")
async def bad_answer_process(request: Request, data: AnswerData):
    """
    Обработчик POST-запросов для записи некорректных ответов ассистента.

    Args:
    - request (Request): Объект запроса FastAPI, содержащий информацию о запросе,
        такую как куки пользователя.
    - data (Data): Экземпляр класса Data, содержащий данные о пользовательском сообщении
        и ответе ассистента.

    Returns:
    - dict: Словарь с единственным ключом "message" и значением "Данные успешно получены".
    """
    user_id = request.cookies.get("user_id") or "anonymous"

    async with aiofiles.open("bad_answers.txt", "a", encoding="utf-8") as file:
        await file.write(f"User ID: {user_id}\n")
        await file.write(f"User: {data.user}\n")
        await file.write(f"Assistant: {data.assistant}\n")
        await file.write("\n")
    return {"message": "Данные успешно получены"}


@app.get("/clear_history")
async def clear_history(request: Request) -> RedirectResponse:
    """
    Очищает историю чата и удаленные файлы синтеза речи для данного пользователя.

    Args:
        request (Request): Объект запроса FastAPI.

    Returns:
        RedirectResponse: Перенаправление на домашнюю страницу.
    """
    session_id = request.cookies.get("user_id")

    file_list = os.listdir(synthesis_path)
    await remove_files(files=file_list)
    # Очистка истории чата для данного пользователя
    chat_history_by_user[session_id] = []
    return RedirectResponse("/", status_code=303)


class ConnectionManager:
    def __init__(self):
        """
        Инициализирует менеджер подключений.
        """
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Устанавливает соединение с WebSocket и добавляет его в активные соединения.

        Args:
            websocket (WebSocket): WebSocket соединение.
            client_id (str): Уникальный идентификатор клиента.
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        """
        Отключает WebSocket соединение и удаляет его из активных соединений.

        Args:
            client_id (str): Уникальный идентификатор клиента.
        """
        del self.active_connections[client_id]

    async def left_chat(self, client_id: str):
        """
        Отправляет сообщение о том, что клиент покинул чат, всем активным соединениям.

        Args:
            client_id (str): Уникальный идентификатор клиента.
        """
        formatted_message = f"""<div class="system-message-content">#{client_id} вышел из чата.</div>"""
        for connection in self.active_connections.values():
            await connection.send_text(formatted_message)

    async def broadcast(self, message: str, sender: str):
        """
        Рассылает сообщение всем активным соединениям, предварительно форматируя его.

        Args:
            message (str): Сообщение, которое нужно отправить.
            sender (str): Имя отправителя сообщения.
        """
        formatted_message = f"""
            <span class="avatar">
                <img src="static/user.png" alt="Avatar">
            </span>
            <div class="info-user">
                <div class="name-user">{sender}</div>
                <div class="message-content">{message}</div>
            </div>
        """
        for connection in self.active_connections.values():
            await connection.send_text(formatted_message)

    def get_active_connections_count(self) -> int:
        """
        Возвращает количество активных подключений.

        Returns:
            int: Количество активных подключений.
        """
        return len(self.active_connections)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Обрабатывает WebSocket подключение.

    Args:
        websocket (WebSocket): WebSocket соединение.
        client_id (str): Уникальный идентификатор клиента.
    """
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(data, client_id)  # Передаем client_id как имя пользователя
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.left_chat(client_id)


@app.get("/chat", response_class=HTMLResponse)
async def get(request: Request):
    """
    Возвращает HTML страницу для чата.

    Args:
        request (Request): Объект запроса FastAPI.

    Returns:
        TemplateResponse: HTML страница для чата.
    """
    session_id = request.cookies.get("user_id") or "anonymous"
    return templates.TemplateResponse("chat.html", {"request": request, "session_id": session_id})


@app.get("/active_connections_count")
async def get_active_connections_count():
    return {"active_connections_count": manager.get_active_connections_count()}


@app.get("/info", response_class=HTMLResponse)
async def read_another_page(request: Request):
    """
    Возвращает HTML страницу для *подробнее.
    """
    return templates.TemplateResponse("info.html", {"request": request})


def main():
    """
    Основная функция приложения.

    Эта функция предназначена для запуска приложения.
    """
    pass


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)