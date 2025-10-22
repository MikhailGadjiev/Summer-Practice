
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import math

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class TimeSeriesForecaster:
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mae, mse, rmse
    
    def prepare_data(self, df):
        """Подготовка и очистка данных"""
        df['dt'] = pd.to_datetime(df['dt'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.sort_values('dt').reset_index(drop=True)
        df['value'] = df['value'].interpolate(method='linear')
        return df
    
    def plot_to_base64(self):
        """Создание графика и преобразование в base64"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graphic = base64.b64encode(image_png).decode('utf-8')
        plt.close()
        return graphic

    def forecast_arima(self, data, periods, train_size):
        """ARIMA прогноз"""
        try:
            # Валидация
            train, test = data[:train_size], data[train_size:]
            arima_model_val = ARIMA(train, order=(5,1,0))
            arima_fit_val = arima_model_val.fit()
            arima_val_forecast = arima_fit_val.forecast(steps=len(test))
            arima_mae, arima_mse, arima_rmse = self.calculate_metrics(test, arima_val_forecast)
            
            # Финальный прогноз
            arima_model_full = ARIMA(data, order=(5,1,0))
            arima_fit_full = arima_model_full.fit()
            arima_forecast_result = arima_fit_full.get_forecast(steps=periods)
            arima_forecast = arima_forecast_result.predicted_mean
            arima_ci = arima_forecast_result.conf_int(alpha=0.05)
            
            # Создаем будущие даты
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                # Определяем частоту данных
                if len(data) > 1:
                    freq = data.index[1] - data.index[0]
                else:
                    freq = pd.Timedelta(days=1)
                
                future_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
            else:
                future_dates = range(len(data), len(data) + periods)
            
            # Визуализация с правильными датами
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, label='Исторические данные', linewidth=2)
            plt.plot(future_dates, arima_forecast, label='ARIMA Прогноз', linewidth=2, color='red')
            plt.fill_between(future_dates, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], 
                          color='red', alpha=0.2, label='ARIMA 95% CI')
            plt.axvline(x=data.index[-1], color='gray', linestyle='--', alpha=0.7, label='Начало прогноза')
            plt.title(f'ARIMA: Прогноз на {periods} периодов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Форматирование дат для лучшего отображения
            if isinstance(last_date, pd.Timestamp):
                plt.gcf().autofmt_xdate()  # Автоповорот дат
            
            plot_url = self.plot_to_base64()
            
            return {
                'success': True,
                'metrics': {'mae': float(arima_mae), 'mse': float(arima_mse), 'rmse': float(arima_rmse)},
                'forecast': arima_forecast.tolist(),
                'ci_lower': arima_ci.iloc[:, 0].tolist(),
                'ci_upper': arima_ci.iloc[:, 1].tolist(),
                'plot': plot_url
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def forecast_prophet(self, data, periods, train_size):
        """Prophet прогноз"""
        try:
            # Валидация
            train, test = data[:train_size], data[train_size:]
            prophet_train = train.reset_index()
            prophet_train.columns = ['ds', 'y']
            prophet_model_val = Prophet()
            prophet_model_val.fit(prophet_train)
            future_val = prophet_model_val.make_future_dataframe(periods=len(test))
            prophet_val_forecast = prophet_model_val.predict(future_val)
            prophet_test_forecast = prophet_val_forecast[prophet_val_forecast['ds'].isin(test.index)]['yhat'].values
            prophet_mae, prophet_mse, prophet_rmse = self.calculate_metrics(test, prophet_test_forecast)
            
            # Финальный прогноз
            prophet_data = data.reset_index()
            prophet_data.columns = ['ds', 'y']
            prophet_model = Prophet(interval_width=0.95)
            prophet_model.fit(prophet_data)
            future = prophet_model.make_future_dataframe(periods=periods)
            prophet_forecast = prophet_model.predict(future)
            
            future_forecast = prophet_forecast[-periods:]
            prophet_future_values = future_forecast['yhat'].values
            prophet_ci_lower = future_forecast['yhat_lower'].values
            prophet_ci_upper = future_forecast['yhat_upper'].values
            
            # Создаем будущие даты для визуализации
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                if len(data) > 1:
                    freq = data.index[1] - data.index[0]
                else:
                    freq = pd.Timedelta(days=1)
                future_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
            else:
                future_dates = range(len(data), len(data) + periods)
            
            # Визуализация с правильными датами
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, label='Исторические данные', linewidth=2)
            plt.plot(future_dates, prophet_future_values, label='Prophet Прогноз', linewidth=2, color='green')
            plt.fill_between(future_dates, prophet_ci_lower, prophet_ci_upper, 
                          color='green', alpha=0.2, label='Prophet 95% CI')
            plt.axvline(x=data.index[-1], color='gray', linestyle='--', alpha=0.7, label='Начало прогноза')
            plt.title(f'Prophet: Прогноз на {periods} периодов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if isinstance(last_date, pd.Timestamp):
                plt.gcf().autofmt_xdate()
            
            plot_url = self.plot_to_base64()
            
            return {
                'success': True,
                'metrics': {'mae': float(prophet_mae), 'mse': float(prophet_mse), 'rmse': float(prophet_rmse)},
                'forecast': prophet_future_values.tolist(),
                'ci_lower': prophet_ci_lower.tolist(),
                'ci_upper': prophet_ci_upper.tolist(),
                'plot': plot_url
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def forecast_holt(self, data, periods, train_size):
        """Holt-Winters прогноз"""
        try:
            # Валидация
            train, test = data[:train_size], data[train_size:]
            hw_model_val = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False)
            hw_fit_val = hw_model_val.fit()
            hw_val_forecast = hw_fit_val.forecast(len(test))
            hw_mae, hw_mse, hw_rmse = self.calculate_metrics(test, hw_val_forecast)
            
            # Финальный прогноз
            hw_model_full = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=False)
            hw_fit_full = hw_model_full.fit()
            hw_forecast = hw_fit_full.forecast(periods)
            
            # Доверительный интервал
            residuals = hw_fit_full.resid.dropna()
            std_residuals = np.std(residuals) if len(residuals) > 0 else np.std(data) * 0.1
            hw_ci_lower = hw_forecast - 1.96 * std_residuals
            hw_ci_upper = hw_forecast + 1.96 * std_residuals
            
            # Создаем будущие даты
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                if len(data) > 1:
                    freq = data.index[1] - data.index[0]
                else:
                    freq = pd.Timedelta(days=1)
                future_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
            else:
                future_dates = range(len(data), len(data) + periods)
            
            # Визуализация с правильными датами
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, label='Исторические данные', linewidth=2)
            plt.plot(future_dates, hw_forecast, label='Holt Прогноз', linewidth=2, color='orange')
            plt.fill_between(future_dates, hw_ci_lower, hw_ci_upper, 
                          color='orange', alpha=0.2, label='Holt 95% CI')
            plt.axvline(x=data.index[-1], color='gray', linestyle='--', alpha=0.7, label='Начало прогноза')
            plt.title(f'Holt: Прогноз на {periods} периодов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if isinstance(last_date, pd.Timestamp):
                plt.gcf().autofmt_xdate()
            
            plot_url = self.plot_to_base64()
            
            return {
                'success': True,
                'metrics': {'mae': float(hw_mae), 'mse': float(hw_mse), 'rmse': float(hw_rmse)},
                'forecast': hw_forecast.tolist(),
                'ci_lower': hw_ci_lower.tolist(),
                'ci_upper': hw_ci_upper.tolist(),
                'plot': plot_url
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def forecast_lstm(self, data, periods, train_size, look_back=10):
        """LSTM прогноз"""
        try:
            # Масштабирование
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
            
            def create_dataset(dataset, look_back=1):
                X, y = [], []
                for i in range(look_back, len(dataset)):
                    X.append(dataset[i-look_back:i, 0])
                    y.append(dataset[i, 0])
                return np.array(X), np.array(y)
            
            # Валидация
            train_scaled = scaled_data[:train_size]
            test_scaled = scaled_data[train_size:]
            
            X_train, y_train = create_dataset(train_scaled, look_back)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            # Создание модели
            def create_lstm_model():
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                return model
            
            lstm_model_val = create_lstm_model()
            lstm_model_val.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Прогноз для валидации
            combined_for_test = np.concatenate([train_scaled[-look_back:], test_scaled])
            test_sequences = []
            for i in range(len(combined_for_test) - look_back):
                test_sequences.append(combined_for_test[i:i + look_back])
            
            if test_sequences:
                X_test = np.array(test_sequences)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                lstm_val_predictions = lstm_model_val.predict(X_test, verbose=0)
                lstm_val_predictions = scaler.inverse_transform(lstm_val_predictions.reshape(-1, 1)).flatten()
                y_test_actual = data[train_size + look_back:train_size + look_back + len(lstm_val_predictions)].values
                min_length = min(len(y_test_actual), len(lstm_val_predictions))
                if min_length > 0:
                    lstm_mae, lstm_mse, lstm_rmse = self.calculate_metrics(
                        y_test_actual[:min_length], lstm_val_predictions[:min_length])
                else:
                    lstm_mae, lstm_mse, lstm_rmse = None, None, None
            else:
                lstm_mae, lstm_mse, lstm_rmse = None, None, None
            
            # Финальный прогноз
            X_full, y_full = create_dataset(scaled_data, look_back)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)
            
            lstm_model_full = create_lstm_model()
            lstm_model_full.fit(X_full, y_full, batch_size=32, epochs=50, verbose=0)
            
            # Прогнозирование в будущее
            lstm_forecast = []
            current_batch = scaled_data[-look_back:].reshape(1, look_back, 1)
            
            for i in range(periods):
                current_pred = lstm_model_full.predict(current_batch, verbose=0)
                lstm_forecast.append(current_pred[0, 0])
                current_batch = np.append(current_batch[:, 1:, :], [[[current_pred[0, 0]]]], axis=1)
            
            lstm_forecast = np.array(lstm_forecast).reshape(-1, 1)
            lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()
            
            # Доверительный интервал
            lstm_ci_lower = lstm_forecast - 0.1 * np.abs(lstm_forecast)
            lstm_ci_upper = lstm_forecast + 0.1 * np.abs(lstm_forecast)
            
            # Создаем будущие даты
            last_date = data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                if len(data) > 1:
                    freq = data.index[1] - data.index[0]
                else:
                    freq = pd.Timedelta(days=1)
                future_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
            else:
                future_dates = range(len(data), len(data) + periods)
            
            # Визуализация с правильными датами
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data.values, label='Исторические данные', linewidth=2)
            plt.plot(future_dates, lstm_forecast, label='LSTM Прогноз', linewidth=2, color='purple')
            plt.fill_between(future_dates, lstm_ci_lower, lstm_ci_upper, 
                          color='purple', alpha=0.2, label='LSTM 95% CI')
            plt.axvline(x=data.index[-1], color='gray', linestyle='--', alpha=0.7, label='Начало прогноза')
            plt.title(f'LSTM: Прогноз на {periods} периодов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if isinstance(last_date, pd.Timestamp):
                plt.gcf().autofmt_xdate()
            
            plot_url = self.plot_to_base64()
            
            return {
                'success': True,
                'metrics': {
                    'mae': float(lstm_mae) if lstm_mae is not None else None,
                    'mse': float(lstm_mse) if lstm_mse is not None else None,
                    'rmse': float(lstm_rmse) if lstm_rmse is not None else None
                },
                'forecast': lstm_forecast.tolist(),
                'ci_lower': lstm_ci_lower.tolist(),
                'ci_upper': lstm_ci_upper.tolist(),
                'plot': plot_url
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

forecaster = TimeSeriesForecaster()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Получение файла
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Получение параметров
        periods = int(request.form.get('periods', 30))
        
        # Чтение файла
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Проверка данных
        if 'dt' not in df.columns or 'value' not in df.columns:
            return jsonify({'error': 'File must contain "dt" and "value" columns'}), 400
        
        # Подготовка данных
        df = forecaster.prepare_data(df)
        data = df.set_index('dt')['value']
        
        # Разделение на train/test
        train_size = int(len(data) * 0.8)
        
        # Прогнозирование всеми моделями
        models = {
            'ARIMA': forecaster.forecast_arima,
            'PROPHET': forecaster.forecast_prophet, 
            'HOLT': forecaster.forecast_holt,
            'LSTM': forecaster.forecast_lstm
        }
        
        results = {}
        for model_name, model_func in models.items():
            print(f"Processing {model_name}...")
            result = model_func(data, periods, train_size)
            results[model_name] = result
        
        # Определение лучшей модели (исключая None значения)
        best_model = None
        best_rmse = float('inf')
        for model_name, result in results.items():
            if (result['success'] and 
                result['metrics']['rmse'] is not None and 
                result['metrics']['rmse'] < best_rmse):
                best_rmse = result['metrics']['rmse']
                best_model = model_name
        
        # Если все модели вернули None, выбираем первую успешную
        if best_model is None:
            for model_name, result in results.items():
                if result['success']:
                    best_model = model_name
                    break
        
        return jsonify({
            'success': True,
            'results': results,
            'best_model': best_model,
            'data_preview': df.head(10).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    try:
        data = request.json
        results = data.get('results', {})
        periods = data.get('periods', 30)
        
        # Создание DataFrame с результатами
        forecast_data = []
        for i in range(periods):
            row = {'period': i + 1}
            for model_name, result in results.items():
                if result.get('success'):
                    row[f'{model_name}_forecast'] = result['forecast'][i] if i < len(result['forecast']) else None
                    row[f'{model_name}_ci_lower'] = result['ci_lower'][i] if i < len(result['ci_lower']) else None
                    row[f'{model_name}_ci_upper'] = result['ci_upper'][i] if i < len(result['ci_upper']) else None
            forecast_data.append(row)
        
        df_results = pd.DataFrame(forecast_data)
        
        # Сохранение в Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Прогнозы', index=False)
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='forecast_results.xlsx'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
