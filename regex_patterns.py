"""
Модуль с регулярными выражениями для обнаружения персональных данных
в русских и казахских текстах
"""

import re

# Шаблоны для обнаружения телефонных номеров (включая российские и казахстанские форматы)
PHONE_PATTERNS = [
    r'\+?[7|8][\s\-\(]?\d{3}[\s\-\)]?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',  # +7(XXX)XXX-XX-XX, 8XXXXXXXXXX
    r'\+?7[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}',  # +7 XXX XXX XXXX
    r'\(\d{3,5}\)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',  # (XXX)XXX-XX-XX
    r'\d{3}[\s\-]?\d{2}[\s\-]?\d{2}',  # XXX-XX-XX (короткие локальные номера)
    r'\d{3,4}[\s\-]?\d{2,3}[\s\-]?\d{2,3}',  # Другие форматы номеров
    r'8\-\d{3}\-\d{3}\-\d{2}\-\d{2}',  # 8-XXX-XXX-XX-XX
    r'8\-\d{3}\-\d{2}\-\d{2}',  # 8-XXX-XX-XX
    # Шаблоны для обнаружения частично замененных данных
    r'[\+]?[7|8]\s*\[\w+\s+\w+\][^а-яА-Я0-9]*',  # +7 [Номер телефона] или вариации с артефактами
    r'[7|8]\s*\[\w+\s+\w+\][^а-яА-Я0-9]*',       # 8 [Номер телефона] или вариации с артефактами
    # Обработка префиксов кодов
    r'\d{3,4}\s+\[\w+\s+\w+\]',                   # 495 [Номер телефона] или другие коды
    # Префикс кода без пробела
    r'\d{3,4}\[\w+\s+\w+\]',                      # 495[Номер телефона]
]

# Шаблоны для обнаружения email-адресов
EMAIL_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # стандартный email
    r'[a-zA-Z0-9._%+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}',  # второй вариант для email
    r'\[\w+-\w+\][а-яА-Яa-zA-Z]*',  # [E-mail] с возможными артефактами
]

# Шаблоны для обнаружения документов
# ИИН (Казахстан)
IIN_PATTERN = r'\b\d{12}\b'
# Паспорт РФ
PASSPORT_PATTERN = r'\b\d{4}[\s\-]?\d{6}\b'
# Серия и номер паспорта РФ в развернутом формате
PASSPORT_EXTENDED_PATTERN = r'(?:паспорт|документ|удостоверение)[\s\:]?\s*\d{2}\s*\d{2}\s*\d{6}'

# Шаблоны для очистки артефактов
CLEANUP_PATTERNS = {
    'phone': r'\[Номер телефона\][а-яА-Яa-zA-Z\[\]]*',
    'email': r'\[E-mail\][а-яА-Яa-zA-Z\[\]]*',
    'name': r'\[Имя\][а-яА-Яa-zA-Z\[\]]*',
    'prefix_phone': r'\d{3,4}\s+\[Номер телефона\]',  # Очистка префиксов типа "495 [Номер телефона]"
    'prefix_phone_nospace': r'\d{3,4}\[Номер телефона\]',  # Очистка префиксов без пробела
}

def get_compiled_patterns():
    """
    Компилирует и возвращает все регулярные выражения
    
    Returns:
        словарь со скомпилированными регулярными выражениями
    """
    
    # Компиляция всех шаблонов телефонных номеров
    compiled_phone_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in PHONE_PATTERNS]
    
    # Компиляция всех шаблонов email-адресов
    compiled_email_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in EMAIL_PATTERNS]
    
    # Компиляция других шаблонов
    compiled_other_patterns = {
        'iin': re.compile(IIN_PATTERN),
        'passport': re.compile(PASSPORT_PATTERN),
        'passport_extended': re.compile(PASSPORT_EXTENDED_PATTERN, re.IGNORECASE),
    }
    
    # Компиляция шаблонов очистки
    compiled_cleanup_patterns = {
        key: re.compile(pattern, re.IGNORECASE) 
        for key, pattern in CLEANUP_PATTERNS.items()
    }
    
    return {
        'phone': compiled_phone_patterns,
        'email': compiled_email_patterns,
        'other': compiled_other_patterns,
        'cleanup': compiled_cleanup_patterns
    }