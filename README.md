🚀 Fraud Detection ML API

Система обнаружения аномалий (мошеннических транзакций) с REST API

📌 О проекте

Проект представляет собой систему для обнаружения аномалий в финансовых транзакциях с использованием методов машинного обучения.

Реализовано:

предобработка и нормализация данных
применение нескольких алгоритмов обнаружения аномалий
оценка качества моделей
REST API для получения предсказаний в реальном времени

⚠️ Датасет содержит анонимизированные признаки (V1–V28), полученные с помощью PCA — это стандартный подход для финансовых данных.

🧠 Используемые методы
Isolation Forest (основной)
Local Outlier Factor (LOF)
DBSCAN

Основной моделью выбран Isolation Forest как наиболее сбалансированный по precision/recall.

📊 Результаты (на 20 000 строках)
Модель	Precision	Recall	F1-score
Isolation Forest	0.5059	0.5059	0.5059
LOF	0.0000	0.0000	0.0000
DBSCAN	0.0300	0.9412	0.0582
🏗️ Структура проекта
fraud-detection-ml/
│
├── data/
│   └── creditcard.csv
│
├── results/
│   └── predictions.xlsx
│
├── src/
│   ├── main.py        # обучение и оценка моделей
│   └── api.py         # FastAPI API
│
├── requirements.txt
└── README.md
⚙️ Установка
git clone https://github.com/fedukliza21-dot/fraud-detection-ml.git
cd fraud-detection-ml

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
▶️ Запуск обучения
python src/main.py

Что происходит:

обучение моделей
расчет метрик
сохранение результатов в Excel
🌐 Запуск API
uvicorn src.api:app --reload

Открыть в браузере:

👉 http://127.0.0.1:8000/docs

📡 Использование API
Endpoint:
POST /predict
Пример запроса
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "...": "...",
  "Amount": 149.62
}
Пример ответа
{
  "anomaly": 0,
  "label": "normal"
}
🧩 Технологии
Python
pandas, numpy
scikit-learn
FastAPI
Uvicorn
🧪 Особенности
полный ML pipeline (от данных до API)
работа с несбалансированными данными
использование unsupervised методов
REST API для инференса
🚀 Возможные улучшения
сохранение модели (joblib)
batch-инференс
логирование и мониторинг
Docker
интеграция с реальными потоками данных
👩‍💻 Автор

Елизавета — студентка ИВТ (МИРЭА), интерес к ML и backend-разработке


выглядит как проект стажёра ✔️
совпадает с требованиями (Python + API + AI) ✔️
показывает мышление, а не просто код ✔️
