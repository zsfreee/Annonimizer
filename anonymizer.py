"""
Модуль для анонимизации персональных данных в тексте
с использованием XLM-RoBERTa и регулярных выражений
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from tqdm import tqdm
import torch
from regex_patterns import get_compiled_patterns

class TextAnonymizer:
    """
    Класс для анонимизации персональных данных в тексте
    с использованием регулярных выражений и XLM-RoBERTa
    """
    
    def __init__(self, models_dir: str = "models", confidence_threshold: float = 0.9,
                 entity_types: List[str] = None):
        """
        Инициализация анонимизатора
        
        Args:
            models_dir: директория для хранения моделей
            confidence_threshold: порог уверенности для NER модели (0-1)
            entity_types: список типов сущностей для анонимизации ['PER', 'ORG', 'LOC', 'MISC']
        """
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        
        # Если типы сущностей не указаны, используем PER, ORG и LOC по умолчанию
        self.entity_types = entity_types or ['PER', 'ORG', 'LOC']
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Получение скомпилированных регулярных выражений
        self.patterns = get_compiled_patterns()
        
        # Флаг для отслеживания загрузки модели
        self.ner_model_loaded = False
        
        # Параметры для модели
        self.ner_model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Используется устройство: {self.device}")
        
    def load_ner_model(self):
        """Загрузка NER-модели для обнаружения именованных сущностей"""
        if not self.ner_model_loaded:
            print(f"Загрузка модели NER ({self.ner_model_name})...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.ner_model_name, 
                    cache_dir=self.models_dir
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.ner_model_name, 
                    cache_dir=self.models_dir
                ).to(self.device)
                
                self.ner_pipeline = pipeline(
                    'ner', 
                    model=self.model, 
                    tokenizer=self.tokenizer, 
                    aggregation_strategy="simple",
                    device=0 if self.device == "cuda" else -1
                )
                
                self.ner_model_loaded = True
                print("NER модель успешно загружена")
            except Exception as e:
                print(f"Ошибка загрузки NER модели: {e}")
                raise

    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста для улучшения работы NER
        
        Args:
            text: исходный текст
        
        Returns:
            обработанный текст
        """
        if not text or isinstance(text, float) or text.isspace():
            return ""
            
        # Преобразование в строку (на случай нестроковых данных)
        text = str(text)
        
        # Добавляем пробелы вокруг знаков препинания для лучшего токенизирования
        text = re.sub(r'([.,!?()[\]{}:;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def find_regex_matches(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Поиск всех совпадений с регулярными выражениями в тексте
        
        Args:
            text: текст для анализа
        
        Returns:
            список кортежей (найденный_текст, начало, конец, тип)
        """
        if not text:
            return []
            
        matches = []
        
        # Поиск телефонных номеров и префиксных кодов
        for pattern in self.patterns['phone']:
            for match in pattern.finditer(text):
                # Проверяем, не пересекается ли с уже найденными совпадениями
                if not self.is_overlap(matches, match.start(), match.end()):
                    # Если это уже частично анонимизированный номер с кодом, 
                    # например "495 [Номер телефона]", добавляем полное совпадение
                    full_match = match.group()
                    if "[Номер телефона]" in full_match:
                        matches.append((full_match, match.start(), match.end(), "[Номер телефона]"))
                    else:
                        matches.append((match.group(), match.start(), match.end(), "[Номер телефона]"))
                
        # Поиск email-адресов
        for pattern in self.patterns['email']:
            for match in pattern.finditer(text):
                if not self.is_overlap(matches, match.start(), match.end()):
                    matches.append((match.group(), match.start(), match.end(), "[E-mail]"))
        
        # Поиск других документов
        for doc_type, pattern in self.patterns['other'].items():
            for match in pattern.finditer(text):
                if not self.is_overlap(matches, match.start(), match.end()):
                    if doc_type == 'iin':
                        matches.append((match.group(), match.start(), match.end(), "[ИИН]"))
                    elif 'passport' in doc_type:
                        matches.append((match.group(), match.start(), match.end(), "[Паспорт]"))
        
        return matches

    def is_overlap(self, matches: List[Tuple[str, int, int, str]], start: int, end: int) -> bool:
        """
        Проверяет, пересекается ли новое совпадение с уже найденными
        
        Args:
            matches: список уже найденных совпадений
            start: начало нового совпадения
            end: конец нового совпадения
            
        Returns:
            True если есть пересечение, False иначе
        """
        for _, match_start, match_end, _ in matches:
            # Проверка на пересечение интервалов
            if max(start, match_start) < min(end, match_end):
                return True
        return False

    def is_already_anonymized(self, text: str, start: int, end: int) -> bool:
        """
        Проверяет, содержит ли фрагмент текста уже анонимизированные данные
        
        Args:
            text: исходный текст
            start: начало фрагмента
            end: конец фрагмента
            
        Returns:
            True если фрагмент содержит метки анонимизации, False иначе
        """
        fragment = text[start:end]
        return any(marker in fragment for marker in ["[Номер телефона]", "[E-mail]", "[Имя]", "[ИИН]", "[Паспорт]", "[Организация]", "[Место]"])

    def find_entities_with_ner(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Поиск именованных сущностей с помощью XLM-RoBERTa
        
        Args:
            text: текст для анализа
            
        Returns:
            список кортежей (найденный_текст, начало, конец, тип)
        """
        if not self.ner_model_loaded:
            self.load_ner_model()
            
        # Проверяем, не пустой ли текст
        if not text or text.isspace():
            return []
        
        try:
            results = self.ner_pipeline(text)
            
            entity_matches = []
            for entity in results:
                # Проверяем уровень уверенности модели
                if entity.get('score', 1.0) < self.confidence_threshold:
                    continue
                    
                # Определяем тип сущности и метку для замены
                entity_type = entity['entity_group']
                replacement = None
                
                # Обрабатываем префиксы B- и I- (начало и продолжение сущности)
                clean_type = entity_type.replace('B-', '').replace('I-', '')
                
                # Проверяем, включен ли этот тип сущности для анонимизации
                if clean_type not in self.entity_types:
                    continue
                    
                # Игнорируем очень короткие сущности (1-2 символа)
                if len(entity['word'].strip()) <= 2 and clean_type in ['PER', 'PERSON']:
                    continue
                    
                # Игнорируем общие слова, которые могут быть ошибочно распознаны как имена
                common_words = ['алло', 'да', 'нет', 'угу', 'вот', 'как', 'где', 'кто', 'что', 'это', 'тоже', 
                               'может', 'быть', 'очень', 'было', 'будет', 'если', 'для', 'при', 'или']
                
                # Проверяем слово в нижнем регистре для игнорирования
                if entity['word'].lower() in common_words:
                    continue
                    
                # Проверяем, есть ли в слове хотя бы одна заглавная буква (признак имени)
                if clean_type in ['PER', 'PERSON'] and not any(c.isupper() for c in entity['word']):
                    # Дополнительная проверка достоверности имени
                    # Если слово не начинается с заглавной буквы и нет контекста обращения, 
                    # вероятно это не имя
                    prev_words = text[:entity['start']].split()[-3:] if entity['start'] > 0 else []
                    name_context = any(w.lower() in ['господин', 'госпожа', 'мистер', 'миссис', 'товарищ', 'гражданин'] 
                                      for w in prev_words)
                                      
                    if not name_context and entity.get('score', 1.0) < 0.95:
                        continue
                
                # Определяем тип сущности
                if clean_type in ['PER', 'PERSON']:
                    replacement = "[Имя]"
                elif clean_type in ['ORG']:
                    replacement = "[Организация]"
                elif clean_type in ['LOC']:
                    replacement = "[Место]"
                elif clean_type == 'MISC':
                    replacement = "[Прочее]"
                else:
                    continue  # Пропускаем другие типы сущностей
                    
                entity_matches.append((
                    entity['word'], 
                    entity['start'], 
                    entity['end'], 
                    replacement
                ))
            
            # Объединяем соседние сущности одного типа
            entity_matches = self.merge_adjacent_entities(entity_matches)
            
            return entity_matches
        except Exception as e:
            print(f"Ошибка при обработке текста NER-моделью: {e}")
            return []

    def merge_adjacent_entities(self, entities: List[Tuple[str, int, int, str]]) -> List[Tuple[str, int, int, str]]:
        """
        Объединение соседних сущностей одного типа
        
        Args:
            entities: список кортежей (текст, начало, конец, тип)
        
        Returns:
            список объединенных сущностей
        """
        if not entities:
            return []
        
        # Сортировка по начальной позиции
        sorted_entities = sorted(entities, key=lambda x: x[1])
        
        merged = []
        current = None
        
        for entity in sorted_entities:
            text, start, end, entity_type = entity
            
            if current is None:
                current = entity
            elif (current[3] == entity_type and  # Одинаковый тип
                  start <= current[2] + 3):      # Находятся рядом (до 3 символов между ними)
                # Объединяем сущности
                new_text = current[0] + text[max(0, start - current[2]):]
                current = (new_text, current[1], end, entity_type)
            else:
                merged.append(current)
                current = entity
        
        if current:
            merged.append(current)
        
        return merged

    def clean_anonymized_text(self, text: str) -> str:
        """
        Очистка анонимизированного текста от возможных артефактов
        
        Args:
            text: анонимизированный текст с возможными артефактами
        
        Returns:
            очищенный анонимизированный текст
        """
        # Очистка артефактов с помощью скомпилированных шаблонов
        cleaned_text = text
        
        # Очистка артефактов телефонов
        cleaned_text = self.patterns['cleanup']['phone'].sub('[Номер телефона]', cleaned_text)
        
        # Очистка артефактов email
        cleaned_text = self.patterns['cleanup']['email'].sub('[E-mail]', cleaned_text)
        
        # Очистка артефактов имен
        cleaned_text = self.patterns['cleanup']['name'].sub('[Имя]', cleaned_text)
        
        # Очистка префиксов кодов телефонов (например, "495 [Номер телефона]")
        cleaned_text = self.patterns['cleanup']['prefix_phone'].sub('[Номер телефона]', cleaned_text)
        cleaned_text = self.patterns['cleanup']['prefix_phone_nospace'].sub('[Номер телефона]', cleaned_text)
        
        # Дополнительная очистка для частично обработанных телефонов
        cleaned_text = re.sub(r'\+?\s?[7|8]\s+\[\w+\s+\w+\]', '[Номер телефона]', cleaned_text)
        cleaned_text = re.sub(r'[7|8]\s+\[\w+\s+\w+\]', '[Номер телефона]', cleaned_text)
        
        # Очистка цифровых префиксов перед номером телефона (повторная, с помощью regex)
        cleaned_text = re.sub(r'\d{3,4}\s*\[Номер телефона\]', '[Номер телефона]', cleaned_text)
        
        return cleaned_text
    
    def process_text_fragment(self, text: str) -> str:
        """
        Обработка фрагмента текста для анонимизации
        
        Args:
            text: текст для обработки
            
        Returns:
            анонимизированный текст
        """
        # Приоритетно ищем телефоны и email
        regex_matches = self.find_regex_matches(text)
        
        # Затем ищем имена с помощью NER
        processed_text = self.preprocess_text(text)
        ner_matches = self.find_entities_with_ner(processed_text) if processed_text else []
        
        # Коррекция позиций NER-сущностей
        corrected_ner_matches = []
        for match_text, _, _, replacement in ner_matches:
            # Ищем соответствие в исходном тексте
            start_pos = text.find(match_text)
            while start_pos != -1:
                if not self.is_already_anonymized(text, start_pos, start_pos + len(match_text)) and \
                   not self.is_overlap(regex_matches + corrected_ner_matches, start_pos, start_pos + len(match_text)):
                    corrected_ner_matches.append((
                        match_text,
                        start_pos,
                        start_pos + len(match_text),
                        replacement
                    ))
                start_pos = text.find(match_text, start_pos + 1)
        
        # Объединяем найденные соответствия и сортируем
        all_matches = regex_matches + corrected_ner_matches
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Применяем замены
        anonymized_text = text
        for match_text, start, end, replacement in all_matches:
            if not self.is_already_anonymized(anonymized_text, start, end):
                anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
        
        # Дополнительная очистка
        anonymized_text = self.clean_anonymized_text(anonymized_text)
        return anonymized_text

    def anonymize_text(self, text: str) -> str:
        """
        Анонимизация персональных данных в тексте
        
        Args:
            text: текст для анонимизации
            
        Returns:
            анонимизированный текст
        """
        if not text or isinstance(text, float) or text.isspace():
            return text
            
        # Для диалогового текста обработаем каждую строку отдельно
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Проверяем, является ли строка частью диалога (Клиент: или Менеджер:)
            if "Клиент:" in line or "Менеджер:" in line:
                # Разделяем строку на префикс и содержимое
                parts = line.split(':', 1)
                if len(parts) == 2:
                    prefix = parts[0] + ":"
                    content = parts[1].strip()
                    # Обрабатываем только содержимое
                    processed_content = self.process_text_fragment(content)
                    processed_line = f"{prefix} {processed_content}"
                else:
                    # Если не удалось разделить - обрабатываем всю строку
                    processed_line = self.process_text_fragment(line)
            else:
                # Для строк не в формате диалога обрабатываем всю строку
                processed_line = self.process_text_fragment(line)
            
            processed_lines.append(processed_line)
        
        # Собираем обработанные строки обратно в текст
        anonymized_text = '\n'.join(processed_lines)
        
        # Финальная очистка для удаления возможных артефактов
        anonymized_text = self.clean_anonymized_text(anonymized_text)
        
        # Дополнительная проверка на префиксы телефонов
        if re.search(r'\d{3,4}\s*\[Номер телефона\]', anonymized_text):
            anonymized_text = re.sub(r'\d{3,4}\s*\[Номер телефона\]', '[Номер телефона]', anonymized_text)
        
        return anonymized_text

    def process_csv(self, input_file: str, output_file: str, text_column: str = "Транскрибация", 
                    anon_column: str = "Анонимизация", callback=None) -> str:
        """
        Обработка CSV-файла с анонимизацией текстов
        
        Args:
            input_file: путь к входному CSV-файлу
            output_file: путь к выходному CSV-файлу
            text_column: название колонки с текстом для анонимизации
            anon_column: название колонки для анонимизированного текста
            callback: функция обратного вызова для обновления прогресса
            
        Returns:
            сообщение о результате обработки
        """
        try:
            # Загрузка CSV-файла с учетом кавычек для мультистрочных ячеек
            print(f"Загрузка файла: {input_file}")
            df = pd.read_csv(input_file, encoding='utf-8', quotechar='"')
            
            # Проверка наличия нужной колонки
            if text_column not in df.columns:
                return f"Ошибка: колонка '{text_column}' не найдена в файле"
                
            # Создание новой колонки для анонимизированного текста, если ее еще нет
            if anon_column not in df.columns:
                df[anon_column] = ""
            
            # Анонимизация текстов
            total_rows = len(df)
            print(f"Всего строк для обработки: {total_rows}")
            
            # Предзагрузка модели перед циклом
            self.load_ner_model()
            
            # Обработка каждой строки
            for i, row in tqdm(df.iterrows(), total=total_rows, desc="Анонимизация"):
                # Получение текста для анонимизации - принудительно преобразуем в строку
                text = str(row[text_column])
                
                # Анонимизация текста
                anonymized_text = self.anonymize_text(text)
                
                # Если после анонимизации текст не изменился, применим более прямой подход
                if anonymized_text == text:
                    # Проверяем наличие регулярных выражений для телефонов и email
                    regex_matches = self.find_regex_matches(text)
                    if regex_matches:
                        # Если найдены потенциальные личные данные, принудительно заменяем их
                        for match_text, start, end, replacement in sorted(regex_matches, key=lambda x: x[1], reverse=True):
                            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
                        # Очистка после принудительной замены
                        anonymized_text = self.clean_anonymized_text(anonymized_text)
                
                # Дополнительная очистка для удаления возможных артефактов
                anonymized_text = self.clean_anonymized_text(anonymized_text)
                
                # Третья проверка для сложных случаев
                if re.search(r'\d{3,4}\s*\[Номер телефона\]', anonymized_text):
                    anonymized_text = re.sub(r'\d{3,4}\s*\[Номер телефона\]', '[Номер телефона]', anonymized_text)
                
                # Сохранение анонимизированного текста
                df.at[i, anon_column] = anonymized_text
                
                # Обновление прогресса, если есть функция обратного вызова
                if callback and i % 10 == 0:
                    progress = int((i + 1) / total_rows * 100)
                    callback(progress)
            
            # Сохранение результата в новый файл с кавычками для мультистрочных ячеек
            df.to_csv(output_file, index=False, encoding='utf-8', quotechar='"')
            
            message = f"Обработка завершена! Анонимизировано {total_rows} записей."
            print(message)
            return message
            
        except Exception as e:
            error_message = f"Ошибка при обработке файла: {e}"
            print(error_message)
            return error_message