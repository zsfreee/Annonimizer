"""
Скрипт для создания тестовых данных CSV с транскрибациями телефонных разговоров
"""

import pandas as pd
import random
import os
import argparse

def generate_test_data(file_path, num_rows=20, include_kaz=True):
    """
    Генерация тестового CSV файла с транскрибациями
    
    Args:
        file_path: путь к сохраняемому CSV файлу
        num_rows: количество строк в файле
        include_kaz: включать ли казахские тексты
    """
    # Примеры телефонных номеров в разных форматах
    phone_formats = [
        "+7 (911) 123-45-67",
        "89271234567",
        "+77017654321",
        "8 (495) 987-65-43",
        "495 765 43 21",
        "+7-916-555-44-33"
    ]
    
    # Примеры email-адресов
    email_formats = [
        "user@example.com",
        "ivan.petrov@mail.ru",
        "test_user@gmail.com",
        "manager@company.kz",
        "support@site.org"
    ]
    
    # Примеры имен (русские)
    ru_names = [
        "Иван Петров",
        "Мария Иванова",
        "Алексей Смирнов",
        "Дмитрий Козлов",
        "Елена Соколова",
        "Сергей Павлов",
        "Ольга Морозова"
    ]
    
    # Примеры имен (казахские)
    kz_names = [
        "Нурлан Сатпаев",
        "Айгуль Бекова",
        "Жанна Ташенова",
        "Арман Касымов",
        "Айжан Нурлатова",
        "Бахыт Пернебаев",
        "Гульнара Ахметова"
    ]
    
    # Объединяем имена
    names = ru_names + kz_names if include_kaz else ru_names
    
    # Шаблоны для русских транскрибаций
    ru_templates = [
        "Здравствуйте, меня зовут {name}, я хотел бы узнать о статусе моего заказа.",
        "Добрый день! Подскажите, как я могу связаться с менеджером? Мой телефон {phone}.",
        "Я бы хотел оставить жалобу. Меня зовут {name}, мой email: {email}.",
        "Добрый день, это {name}. Не могли бы вы перезвонить мне по номеру {phone}?",
        "Здравствуйте! Я отправил вам документы на почту {email}, проверьте пожалуйста.",
        "Меня зовут {name}, я хочу уточнить детали доставки на мой адрес.",
        "Я не могу зайти в личный кабинет. Мой логин {email}, помогите пожалуйста.",
        "Добрый день, мне нужна консультация. Перезвоните на {phone}, спросите {name}."
    ]
    
    # Шаблоны для казахских транскрибаций
    kz_templates = [
        "Сәлеметсіз бе! Менің атым {name}. Тапсырысым қашан жеткізіледі?",
        "Мен сізге {email} поштасына хат жібердім, қарап шығыңызшы.",
        "Сәлем! Бұл {name}, маған {phone} нөміріне хабарласыңызшы.",
        "Қайырлы күн! Мен {name}, тауарды ауыстырғым келеді.",
        "Сәлеметсіз бе! Менің паролімді қалпына келтіруге көмектесіңізші. Менің поштам {email}.",
        "Кеңесші қызметіне қосылғым келеді. Менің нөмірім {phone}."
    ]
    
    # Объединяем шаблоны
    templates = ru_templates + kz_templates if include_kaz else ru_templates
    
    # Генерация данных
    data = []
    for i in range(num_rows):
        call_id = f"CALL-{random.randint(10000, 99999)}"
        
        # Выбор случайного шаблона
        template = random.choice(templates)
        
        # Заполнение шаблона данными
        name = random.choice(names)
        phone = random.choice(phone_formats)
        email = random.choice(email_formats)
        
        transcription = template.format(name=name, phone=phone, email=email)
        
        # Иногда добавляем дополнительные данные
        if random.random() < 0.3:
            additional_name = random.choice(names)
            transcription += f" Также вы можете связаться с {additional_name} по этому вопросу."
        
        if random.random() < 0.3:
            additional_phone = random.choice(phone_formats)
            transcription += f" Мой альтернативный номер: {additional_phone}."
            
        # Добавляем запись в данные
        data.append({
            "ID": call_id,
            "Дата": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "Длительность": f"{random.randint(1, 15)}:{random.randint(0, 59):02d}",
            "Транскрибация": transcription
        })
    
    # Создание датафрейма и сохранение в CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding='utf-8')
    
    print(f"Тестовый CSV файл с {num_rows} записями сохранен в {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация тестовых данных для анонимизатора')
    parser.add_argument('-o', '--output', default='test_data.csv', 
                        help='Путь к выходному CSV файлу')
    parser.add_argument('-n', '--num-rows', type=int, default=50, 
                        help='Количество строк для генерации')
    parser.add_argument('--no-kaz', action='store_true', 
                        help='Не включать казахские тексты')
    
    args = parser.parse_args()
    
    # Создание директории для файла, если необходимо
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Генерация тестовых данных
    generate_test_data(args.output, args.num_rows, not args.no_kaz)