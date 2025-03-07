"""
Графический интерфейс для анонимизатора текста
"""

import os
import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QProgressBar, QTextEdit,
    QLineEdit, QGroupBox, QSplitter, QComboBox, QCheckBox,
    QDoubleSpinBox, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
import pandas as pd
import traceback

from anonymizer import TextAnonymizer

class AnonymizerThread(QThread):
    """Поток для запуска анонимизации в фоновом режиме"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(str)
    
    def __init__(self, anonymizer, input_file, output_file, text_column, output_simple_file=None):
        super().__init__()
        self.anonymizer = anonymizer
        self.input_file = input_file
        self.output_file = output_file
        self.text_column = text_column
        self.output_simple_file = output_simple_file
        
    def run(self):
        """Запуск анонимизации в отдельном потоке"""
        try:
            # Основной файл с полными данными
            result = self.anonymizer.process_csv(
                self.input_file, 
                self.output_file, 
                self.text_column,
                callback=self.progress_signal.emit
            )
            
            # Если указан файл для упрощенного вывода, создаем его
            if self.output_simple_file:
                try:
                    # Загружаем обработанный CSV файл
                    df = pd.read_csv(self.output_file, encoding='utf-8', quotechar='"')
                    
                    # Определяем колонку с идентификатором файла/записи
                    id_column = None
                    for candidate in ['ID', 'Файл', 'File', 'id', 'file', 'filename', 'Имя файла']:
                        if candidate in df.columns:
                            id_column = candidate
                            break
                    
                    # Если не нашли подходящую колонку, используем первую
                    if id_column is None:
                        id_column = df.columns[0]
                    
                    # Создаем новый датафрейм только с ID и анонимизацией
                    simple_df = df[[id_column, 'Анонимизация']]
                    
                    # Сохраняем результат
                    simple_df.to_csv(self.output_simple_file, index=False, encoding='utf-8', quotechar='"')
                    result += f"\nСоздан упрощенный файл: {self.output_simple_file}"
                except Exception as e:
                    result += f"\nОшибка при создании упрощенного файла: {e}"
            
            self.result_signal.emit(result)
        except Exception as e:
            error_message = f"Ошибка при обработке файла: {e}\n{traceback.format_exc()}"
            self.result_signal.emit(error_message)

class AnonymizerApp(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.anonymizer = TextAnonymizer()
        self.input_file = ""
        self.output_file = ""
        self.output_simple_file = ""
        self.initUI()
        
    def initUI(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Анонимизатор текста в CSV файлах")
        self.setGeometry(100, 100, 800, 600)
        
        # Установка иконки приложения
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Основной виджет и компоновка
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Группа для выбора файлов
        self.create_file_group()
        
        # Группа для параметров
        self.create_settings_group()
        
        # Группа для предпросмотра
        self.create_preview_group()
        
        # Прогресс-бар и кнопка запуска
        self.create_progress_group()
        
        # Журнал действий
        self.create_log_group()
        
        # Первоначальное обновление интерфейса
        self.update_ui_state()
        
    def create_file_group(self):
        """Создание группы для выбора файлов"""
        file_group = QGroupBox("Выбор файлов")
        file_layout = QVBoxLayout()
        
        # Выбор входного файла
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Входной CSV:")
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        self.input_btn = QPushButton("Выбрать...")
        self.input_btn.clicked.connect(self.select_input_file)
        
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_btn)
        
        # Выбор выходных файлов
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Выходные файлы:")
        file_layout.addLayout(input_layout)
        file_layout.addWidget(self.output_label)
        
        # Полный выходной файл
        full_output_layout = QHBoxLayout()
        self.full_output_label = QLabel("Полные данные:")
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        self.output_btn = QPushButton("Выбрать...")
        self.output_btn.clicked.connect(self.select_output_file)
        
        full_output_layout.addWidget(self.full_output_label)
        full_output_layout.addWidget(self.output_path)
        full_output_layout.addWidget(self.output_btn)
        
        # Упрощенный выходной файл
        simple_output_layout = QHBoxLayout()
        self.simple_output_label = QLabel("Только анонимизация:")
        self.simple_output_path = QLineEdit()
        self.simple_output_path.setReadOnly(True)
        self.simple_output_btn = QPushButton("Выбрать...")
        self.simple_output_btn.clicked.connect(self.select_simple_output_file)
        
        simple_output_layout.addWidget(self.simple_output_label)
        simple_output_layout.addWidget(self.simple_output_path)
        simple_output_layout.addWidget(self.simple_output_btn)
        
        file_layout.addLayout(full_output_layout)
        file_layout.addLayout(simple_output_layout)
        file_group.setLayout(file_layout)
        
        self.main_layout.addWidget(file_group)
    
    def create_settings_group(self):
        """Создание группы для настроек"""
        settings_group = QGroupBox("Настройки")
        settings_layout = QGridLayout()
        
        # Выбор колонки для обработки
        self.column_label = QLabel("Колонка с текстом:")
        self.column_combo = QComboBox()
        self.column_combo.setEnabled(False)
        settings_layout.addWidget(self.column_label, 0, 0)
        settings_layout.addWidget(self.column_combo, 0, 1)
        
        # Порог уверенности для модели NER
        self.confidence_label = QLabel("Порог уверенности:")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.9)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setToolTip("Минимальная оценка уверенности модели (0-1).\n"
                                      "Высокие значения снижают количество ложных срабатываний,\n"
                                      "низкие значения улучшают обнаружение, но могут дать больше ошибок.")
        settings_layout.addWidget(self.confidence_label, 1, 0)
        settings_layout.addWidget(self.confidence_spin, 1, 1)
        
        # Группа чекбоксов для типов сущностей
        entity_group = QGroupBox("Типы сущностей для анонимизации")
        entity_layout = QVBoxLayout()

        self.per_checkbox = QCheckBox("PER - Персоны (имена, фамилии)")
        self.per_checkbox.setChecked(True)  # По умолчанию включено
        self.per_checkbox.setToolTip("Анонимизировать имена и фамилии людей")

        self.org_checkbox = QCheckBox("ORG - Организации")
        self.org_checkbox.setChecked(True)  # По умолчанию включено
        self.org_checkbox.setToolTip("Анонимизировать названия организаций")

        self.loc_checkbox = QCheckBox("LOC - Локации")
        self.loc_checkbox.setChecked(True)  # По умолчанию включено
        self.loc_checkbox.setToolTip("Анонимизировать названия географических объектов")

        self.misc_checkbox = QCheckBox("MISC - Прочие сущности")
        self.misc_checkbox.setChecked(False)  # По умолчанию выключено
        self.misc_checkbox.setToolTip("Анонимизировать прочие типы сущностей (национальности, события и т.д.)")

        entity_layout.addWidget(self.per_checkbox)
        entity_layout.addWidget(self.org_checkbox)
        entity_layout.addWidget(self.loc_checkbox)
        entity_layout.addWidget(self.misc_checkbox)
        entity_group.setLayout(entity_layout)

        settings_layout.addWidget(entity_group, 2, 0, 1, 2)
        
        # Тестовый текст
        self.test_label = QLabel("Тестовый текст:")
        self.test_text = QTextEdit()
        self.test_text.setPlaceholderText("Введите текст для тестовой анонимизации...")
        self.test_btn = QPushButton("Тестировать")
        self.test_btn.clicked.connect(self.test_anonymization)
        
        settings_layout.addWidget(self.test_label, 3, 0)
        settings_layout.addWidget(self.test_text, 4, 0, 1, 2)
        settings_layout.addWidget(self.test_btn, 5, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        self.main_layout.addWidget(settings_group)
    
    def create_preview_group(self):
        """Создание группы для предпросмотра данных"""
        preview_group = QGroupBox("Предпросмотр")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Загрузите CSV файл для предпросмотра...")
        
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        
        self.main_layout.addWidget(preview_group)
    
    def create_progress_group(self):
        """Создание группы для прогресса и кнопок управления"""
        progress_layout = QHBoxLayout()
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Кнопки
        self.start_btn = QPushButton("Запустить анонимизацию")
        self.start_btn.clicked.connect(self.start_anonymization)
        self.start_btn.setEnabled(False)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.start_btn)
        
        self.main_layout.addLayout(progress_layout)
    
    def create_log_group(self):
        """Создание группы для журнала"""
        log_group = QGroupBox("Журнал")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        self.main_layout.addWidget(log_group)
    
    def select_input_file(self):
        """Выбор входного CSV файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбор входного CSV файла", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            self.input_file = file_path
            self.input_path.setText(file_path)
            self.log_message(f"Выбран входной файл: {file_path}")
            
            # Автоматически устанавливаем пути выходных файлов в той же папке
            dir_path = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            
            # Полный файл вывода
            default_output = os.path.join(dir_path, f"{name}_anon{ext}")
            self.output_file = default_output
            self.output_path.setText(default_output)
            
            # Упрощенный файл вывода
            simple_output = os.path.join(dir_path, f"{name}_simple{ext}")
            self.output_simple_file = simple_output
            self.simple_output_path.setText(simple_output)
            
            # Обновляем предпросмотр и список колонок
            self.update_preview()
            self.update_ui_state()
    
    def select_output_file(self):
        """Выбор полного выходного CSV файла"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Выбор полного выходного CSV файла", self.output_file, "CSV Files (*.csv)"
        )
        
        if file_path:
            self.output_file = file_path
            self.output_path.setText(file_path)
            self.log_message(f"Выбран полный выходной файл: {file_path}")
            self.update_ui_state()
    
    def select_simple_output_file(self):
        """Выбор упрощенного выходного CSV файла"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Выбор упрощенного выходного CSV файла", self.output_simple_file, "CSV Files (*.csv)"
        )
        
        if file_path:
            self.output_simple_file = file_path
            self.simple_output_path.setText(file_path)
            self.log_message(f"Выбран упрощенный выходной файл: {file_path}")
            self.update_ui_state()
    
    def update_preview(self):
        """Обновление предпросмотра и списка колонок"""
        try:
            if self.input_file:
                # Загружаем CSV, учитывая кавычки для мультистрочных ячеек
                df = pd.read_csv(self.input_file, encoding='utf-8', quotechar='"')
                
                # Обновляем выпадающий список колонок
                self.column_combo.clear()
                for column in df.columns:
                    self.column_combo.addItem(column)
                
                # Выбираем колонку "Транскрибация" по умолчанию, если есть
                default_index = self.column_combo.findText("Транскрибация")
                if default_index >= 0:
                    self.column_combo.setCurrentIndex(default_index)
                
                # Включаем выбор колонки
                self.column_combo.setEnabled(True)
                
                # Обновляем предпросмотр (первые 5 строк)
                preview = df.head(5).to_string()
                self.preview_text.setText(preview)
                
                self.log_message(f"Загружен CSV с {len(df)} строками и {len(df.columns)} колонками")
                
        except Exception as e:
            self.log_message(f"Ошибка при загрузке CSV: {e}")
    
    def update_ui_state(self):
        """Обновление состояния интерфейса в зависимости от выбранных файлов"""
        has_input = bool(self.input_file)
        has_output = bool(self.output_file)
        
        # Включаем кнопку запуска только если выбраны входной и хотя бы один выходной файл
        self.start_btn.setEnabled(has_input and has_output)
    
    def test_anonymization(self):
        """Тестовая анонимизация текста"""
        text = self.test_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Предупреждение", "Введите текст для тестирования")
            return
        
        # Обновляем параметры анонимизатора
        confidence = self.confidence_spin.value()
        self.anonymizer.confidence_threshold = confidence
        
        # Собираем выбранные типы сущностей
        selected_entity_types = []
        if self.per_checkbox.isChecked():
            selected_entity_types.append('PER')
        if self.org_checkbox.isChecked():
            selected_entity_types.append('ORG')
        if self.loc_checkbox.isChecked():
            selected_entity_types.append('LOC')
        if self.misc_checkbox.isChecked():
            selected_entity_types.append('MISC')
            
        self.anonymizer.entity_types = selected_entity_types
        
        # Анонимизация текста
        self.log_message("Тестовая анонимизация:")
        self.log_message(f"Исходный текст: {text}")
        
        try:
            # Принудительно загружаем модель, если еще не загружена
            if not self.anonymizer.ner_model_loaded:
                self.anonymizer.load_ner_model()
                
            anonymized = self.anonymizer.anonymize_text(text)
            self.log_message(f"Анонимизированный текст: {anonymized}")
            
            if anonymized == text:
                self.log_message("⚠️ Внимание: текст не изменился после анонимизации")
            else:
                self.log_message("✅ Текст успешно анонимизирован")
                
        except Exception as e:
            self.log_message(f"❌ Ошибка при анонимизации: {e}")
    
    def start_anonymization(self):
        """Запуск процесса анонимизации"""
        if not self.input_file or not self.output_file:
            self.log_message("Ошибка: не выбраны входной или выходной файлы")
            return
        
        # Получаем выбранную колонку
        selected_column = self.column_combo.currentText()
        if not selected_column:
            self.log_message("Ошибка: не выбрана колонка с текстом")
            return
        
        # Отключаем кнопки на время выполнения
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.simple_output_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.column_combo.setEnabled(False)
        self.test_btn.setEnabled(False)
        
        # Сбрасываем прогресс
        self.progress_bar.setValue(0)
        
        # Обновляем параметры анонимизатора
        self.anonymizer.confidence_threshold = self.confidence_spin.value()
        
        # Собираем выбранные типы сущностей
        selected_entity_types = []
        if self.per_checkbox.isChecked():
            selected_entity_types.append('PER')
        if self.org_checkbox.isChecked():
            selected_entity_types.append('ORG')
        if self.loc_checkbox.isChecked():
            selected_entity_types.append('LOC')
        if self.misc_checkbox.isChecked():
            selected_entity_types.append('MISC')
            
        self.anonymizer.entity_types = selected_entity_types
        
        # Определяем, нужно ли создавать упрощенный файл
        use_simple_file = bool(self.output_simple_file)
        
        # Создаем и запускаем поток
        self.log_message(f"Начало анонимизации. Колонка: {selected_column}, типы сущностей: {', '.join(selected_entity_types)}")
        
        # Включаем информацию о создании файлов в лог
        if use_simple_file:
            self.log_message(f"Будут созданы два файла: полный ({self.output_file}) и упрощенный ({self.output_simple_file})")
        else:
            self.log_message(f"Будет создан только полный файл: {self.output_file}")
        
        # Создаем и запускаем поток анонимизации
        simple_file = self.output_simple_file if use_simple_file else None
        self.thread = AnonymizerThread(
            self.anonymizer, 
            self.input_file, 
            self.output_file, 
            selected_column,
            simple_file
        )
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.result_signal.connect(self.process_completed)
        self.thread.start()
    
    def update_progress(self, value):
        """Обновление индикатора прогресса"""
        self.progress_bar.setValue(value)
    
    def process_completed(self, result):
        """Обработка завершения анонимизации"""
        self.log_message(result)
        self.progress_bar.setValue(100)
        
        # Повторно включаем кнопки
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.simple_output_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.column_combo.setEnabled(True)
        self.test_btn.setEnabled(True)
        
        # Если в результате есть "Ошибка", показываем сообщение об ошибке
        if "Ошибка" in result:
            QMessageBox.critical(self, "Ошибка", result)
        else:
            # Показываем сообщение о завершении без предложения открыть файл
            QMessageBox.information(self, "Обработка завершена", 
                                  "Анонимизация успешно завершена.")
    
    def log_message(self, message):
        """Добавление сообщения в журнал"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        # Автоматическая прокрутка до конца
        self.log_text.moveCursor(self.log_text.textCursor().End)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnonymizerApp()
    window.show()
    sys.exit(app.exec_())
