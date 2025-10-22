
@echo off
chcp 65001 > nul
echo ========================================
echo   Прогнозирование временных рядов
echo ========================================
echo.

echo Проверка установки Python...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Python не установлен или не добавлен в PATH
    echo Скачайте Python с https://python.org
    pause
    exit /b 1
)

echo Установка зависимостей...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ОШИБКА: Не удалось установить зависимости
    pause
    exit /b 1
)

echo.
echo ========================================
echo Запуск приложения...
echo Откройте браузер и перейдите по адресу:
echo http://localhost:5000
echo ========================================
echo.

python app.py

pause
