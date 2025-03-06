# Анонимизатор телефонных транскрибаций

Программа для анонимизации персональных данных (имена, телефоны, email и др.) в CSV-файлах с записями телефонных разговоров.

## Особенности

- Анонимизация персональных данных в CSV-файлах с транскрибациями
- Поддержка русского и казахского языков
- Использование XLM-RoBERTa для распознавания именованных сущностей
- Регулярные выражения для обнаружения телефонов, email и документов
- Графический интерфейс с возможностью выбора типов сущностей для анонимизации
- Сохранение исходных данных и добавление колонки с анонимизированным текстом
- Защита от артефактов при повторной обработке текста

## Требования

- Python 3.8+ (тестировано на 3.12.7)
- PyQt5==5.15.11
- pandas==2.2.3
- transformers==4.49.0
- torch==2.6.0
- regex==2024.11.6
- tqdm==4.67.1
- colorama==0.4.6
- sentencepiece (устанавливается автоматически с transformers)

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/zsfreee/Annonimizer.git
cd anonymizer
```

2. Создайте виртуальное окружение и установите зависимости:

```bash
# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# На Windows:
venv\Scripts\activate
# На macOS/Linux:
# source venv/bin/activate
```
# Установка необходимых пакетов
Не забываем обновить pip
```
python.exe -m pip install --upgrade pip
```
pip install -r requirements.txt
```

## Использование

### Графический интерфейс

Для запуска программы с графическим интерфейсом:

```bash
python main.py
```

1. Нажмите "Выбрать..." для выбора входного CSV-файла
2. Подтвердите или измените путь к выходному CSV-файлу
3. Выберите колонку с текстом (по умолчанию "Транскрибация")
4. Настройте параметры анонимизации:
   - Выберите порог уверенности для модели (0.1-1.0)
   - Отметьте типы сущностей для анонимизации (PER, ORG, LOC, MISC)
5. Нажмите "Запустить анонимизацию"
6. Следите за прогрессом в журнале и прогресс-баре

### Опции анонимизации сущностей

Программа позволяет выбрать, какие типы именованных сущностей следует анонимизировать:

- **PER - Персоны (имена, фамилии)** - по умолчанию включено, заменяет на "[Имя]"
- **ORG - Организации** - по умолчанию включено, заменяет на "[Организация]" 
- **LOC - Локации** - по умолчанию включено, заменяет на "[Место]"
- **MISC - Прочие сущности** - по умолчанию выключено, заменяет на "[Прочее]"

Эта опция позволяет более гибко настраивать процесс анонимизации в зависимости от требований. Например, вы можете анонимизировать только имена людей, сохранив при этом названия организаций или географических объектов.

### Тестирование анонимизации текста

Интерфейс программы включает возможность протестировать анонимизацию на отдельных фрагментах текста:

1. Введите текст в поле "Тестовый текст"
2. Настройте параметры анонимизации (порог уверенности и типы сущностей)
3. Нажмите кнопку "Тестировать"
4. Просмотрите результат в журнале

### Генерация тестовых данных

Для создания тестовых данных:

```bash
python generate_test_data.py -o test_data.csv -n 50
```

Параметры:
- `-o, --output` - путь к выходному CSV-файлу (по умолчанию "test_data.csv")
- `-n, --num-rows` - количество строк для генерации (по умолчанию 50)
- `--no-kaz` - не включать казахские тексты

## Примечания

1. **Первый запуск**:
   - При первом запуске программа загрузит модель NER (~1.2ГБ), что может занять время
   - Последующие запуски будут использовать кэшированную модель из папки `models/`

2. **Производительность**:
   - Если доступен GPU (CUDA), модель будет использовать его для ускорения
   - Большие файлы могут обрабатываться долго, следите за прогрессом

3. **Настройка параметров**:
   - **Порог уверенности** (confidence threshold) - минимальная оценка уверенности модели (от 0 до 1) для принятия результата распознавания. Значение 0.90 (по умолчанию) означает, что будут учитываться только те именованные сущности, в которых модель уверена не менее чем на 90%. Повышение порога снижает количество ложных срабатываний, но может привести к пропуску некоторых сущностей. Понижение порога увеличивает полноту обнаружения, но может привести к ложным срабатываниям.
   - **Типы сущностей** - позволяет выбрать, какие категории сущностей будут анонимизированы. Установка флажков PER, ORG, LOC, MISC определяет, какие типы данных будут распознаваться и заменяться.

4. **Расширение функциональности**:
   - Для добавления новых типов данных для анонимизации, отредактируйте `regex_patterns.py`
   - Для настройки параметров NER-модели, измените соответствующие параметры в `anonymizer.py`

5. **Обработка артефактов**:
   - Программа содержит дополнительные механизмы для предотвращения появления артефактов при анонимизации
   - Если обнаруживаются артефакты вида "[Номер телефона]она]фона]", они автоматически исправляются
   - Двойная обработка текста (анонимизация уже анонимизированного текста) также обрабатывается корректно

## Структура проекта

```
anonymizer_project/
│
├── main.py                # Основной файл с GUI
├── anonymizer.py          # Модуль для анонимизации
├── regex_patterns.py      # Модуль с регулярными выражениями
├── generate_test_data.py  # Скрипт для создания тестовых данных
├── requirements.txt       # Файл с зависимостями
├── README.md              # Документация
│
└── models/                # Папка для кэширования моделей (создается автоматически)
```

## Известные ограничения

1. Модель XLM-RoBERTa может неточно распознавать некоторые казахские имена, особенно редкие
2. Для телефонных номеров в нестандартных форматах может потребоваться дополнительная настройка шаблонов
3. При анонимизации очень больших файлов рекомендуется использовать мощный компьютер с GPU

## Лицензия

MIT
