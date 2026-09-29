# external_adapter/weather.py

import httpx
from typing import Dict, Any
from datetime import datetime, timedelta

class WeatherAdapter:
    """Open-Meteo REST API를 이용해 예보 정보를 조회하는 외부 어댑터 클래스"""

    def __init__(self, base_url: str = "https://api.open-meteo.com/v1/forecast"):
        self.base_url = base_url

    def get_forecast(self, latitude: float = 37.5665, longitude: float = 126.9780) -> Dict[str, Any]:
        """위도, 경도 좌표를 기반으로 실시간 및 일일 날씨 데이터를 조회합니다.
        날씨 정보를 조회하고, 정확한 날짜와 함께 결과를 반환합니다.

        Args:
            latitude (float): 위도 (기본값: 서울 37.5665)
            longitude (float): 경도 (기본값: 서울 126.9780)

        Returns:
            dict: 날씨 데이터 딕셔너리
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "weather_code"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"],
            "timezone": "Asia/Tokyo"
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()