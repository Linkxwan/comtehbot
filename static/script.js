// Функция, которая фокусирует на поле ввода после загрузки страницы
window.onload = function() {
    document.getElementById("question-input").focus();
};

window.onload = function() {
    window.scrollTo(0, document.body.scrollHeight);
};
function validateForm() {
    var questionInput = document.getElementById("question-input").value;
    if (questionInput.trim() === "") {
        return false;
    }
    return true;
}

async function sendQuestion(question) {
    try {
        const formData = new FormData(); // Создаем объект FormData
        formData.append('question', question); // Добавляем текст вопроса

        const response = await fetch('/api', { // Отправляем POST-запрос на сервер
            method: 'POST',
            body: formData // Устанавливаем тело запроса
        });

        if (!response.ok) { // Проверяем успешность запроса
            throw new Error('Ошибка сети');
        }

        // Действия с успешным ответом от сервера (если нужно)

        // Перезагрузка страницы для обновления истории чата
        window.location.reload();
    } catch (error) {
        console.error('Произошла ошибка:', error);
        // Действия при ошибке
    }
}

async function sendSynthesisRequest(index) {
    const messageContent = document.getElementById(`assistant-message-${index}`).textContent;
    console.log(messageContent);
    const response = await fetch(`/synthesis?data=${messageContent}`, {
        method: 'GET',
    });

    if (!response.ok) {
        throw new Error('Ошибка отправки запроса');
    }

    const responseData = await response.json(); // или response.text() в зависимости от формата ответа
    const audioURL = responseData['created_file']
    console.log(audioURL); // вывод данных в консоль
    playAudio(audioURL);
}

function playAudio(audioURL) {
    const audio = new Audio(audioURL);
    audio.play();
}

function copyText(index) {
    const messageContent = document.getElementById(`assistant-message-${index}`).textContent;

    navigator.clipboard.writeText(messageContent)
        .then(() => {
            const notification = document.getElementById('copy-notification');
            notification.style.display = 'block';
            setTimeout(() => {
                notification.style.display = 'none';
            }, 2000); // Скрыть уведомление через 2 секунды
        });
}

function bad_answer(index) {
    const userMessage = document.getElementById(`user-message-${index - 1}`).textContent;
    const assistantMessage = document.getElementById(`assistant-message-${index}`).textContent;

    const data = {
        user: userMessage,
        assistant: assistantMessage
    };

    fetch('/bad_answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Ошибка отправки запроса');
        }
        console.log('Данные успешно отправлены');
    })
    .catch(error => {
        console.error(error);
    });
}

document.getElementById("clear-history").addEventListener("click", function() {
    fetch("/clear_history")
        .then(response => {
            if (response.ok) {
                window.location.reload();
            } else {
                console.error("Ошибка при очистке истории чата");
            }
        });
});
// Функция для добавления эффекта появления текста по буквам
function typeMessage(element) {
    var message = element.textContent;
    element.textContent = ''; // Очищаем содержимое элемента
    var index = 0;
    var typingInterval = setInterval(function() {
        if (index < message.length) {
            element.textContent += message.charAt(index);
            index++;
        } else {
            clearInterval(typingInterval);
        }
    }, 10); // Задержка между появлением букв (в миллисекундах)
}
// Получаем последний элемент с классом "message-content" и вызываем функцию для добавления эффекта появления по буквам
var messageContentElements = document.querySelectorAll('.message-content');
var lastMessageElement = messageContentElements[messageContentElements.length - 1];
if (lastMessageElement) {
    typeMessage(lastMessageElement);
}


// Выбираем textarea
var textarea = document.getElementById("question-input");

textarea.addEventListener("input", function(e) {
  if (e.inputType !== "insertText") return; // Игнорируем события, не связанные с добавлением текста

  // Обновляем высоту textarea
  this.style.height = "auto";
  this.style.height = (this.scrollHeight) + "px";
});

textarea.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && e.shiftKey) {
    var cursorPosition = this.selectionStart; // Определяем позицию курсора
    var textBefore = this.value.substring(0, cursorPosition); // Текст до позиции курсора
    var textAfter = this.value.substring(cursorPosition, this.value.length); // Текст после позиции курсора

    // Объединяем текст до и после курсора с новой строкой между ними
    this.value = textBefore + '\n' + textAfter;

    // Перемещаем курсор на новую строку
    this.selectionStart = this.selectionEnd = cursorPosition + 1;

    this.style.height = "auto"; // Обновляем высоту textarea после добавления новой строки
    this.style.height = (this.scrollHeight) + "px";
    
    e.preventDefault(); // Предотвращаем действие по умолчанию (перенос строки)
  }
});

document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault(); // Предотвращаем действие по умолчанию (в данном случае перенос строки)
        document.getElementById('send-button').click(); // Вызываем нажатие на кнопку
    }
});

textarea.addEventListener("input", function() {
  // Устанавливаем высоту textarea на основе содержимого
  this.style.height = "auto";
  this.style.height = (this.scrollHeight) + "px";
});

// Получаем ссылку на textarea
var textarea = document.getElementById('question-input');

// Добавляем обработчик события input, который срабатывает при изменении содержимого textarea
textarea.addEventListener('input', function() {
    var button = document.getElementById('send-button');
    // Если в textarea есть текст, добавляем кнопке класс active
    if (this.value.trim() !== '') {
        button.classList.add('active');
    } else {
        button.classList.remove('active');
    }
});