from flask import Flask
from typing import List


class CORSManager:
    """CORS管理器"""

    @staticmethod
    def setup_cors(app: Flask, origins: List[str] = None) -> None:
        """设置全局CORS配置"""
        if origins is None:
            origins = ["*"]

        @app.after_request
        def after_request(response):
            """全局CORS配置"""
            response.headers.update({
                'Access-Control-Allow-Origin': ', '.join(origins),
                'Access-Control-Allow-Headers': 'Content-Type,Authorization,Accept,X-Requested-With',
                'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            })
            return response