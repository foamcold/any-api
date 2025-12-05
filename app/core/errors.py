"""
统一的错误处理工具

提供创建和转换各种 API 格式错误响应的工具函数。
支持格式：OpenAI、Gemini
"""

from typing import Dict, Any, Literal
import json
import time
import uuid

ApiFormat = Literal["openai", "gemini"]


class APIError:
    """API 错误响应构造器"""
    
    @staticmethod
    def openai_error(
        message: str,
        error_type: str = "api_error",
        code: str = None,
        param: str = None
    ) -> Dict[str, Any]:
        """
        创建 OpenAI 格式的错误响应
        
        Args:
            message: 错误消息
            error_type: 错误类型（api_error, invalid_request_error, authentication_error, rate_limit_error等）
            code: 错误代码
            param: 相关参数名
            
        Returns:
            OpenAI 格式的错误对象
        """
        return {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    
    @staticmethod
    def gemini_error(
        message: str,
        status: str = "UNKNOWN",
        code: int = None,
        details: list = None
    ) -> Dict[str, Any]:
        """
        创建 Gemini 格式的错误响应
        
        Args:
            message: 错误消息
            status: 错误状态（INVALID_ARGUMENT, PERMISSION_DENIED, UNAUTHENTICATED等）
            code: HTTP 状态码
            details: 详细错误信息列表
            
        Returns:
            Gemini 格式的错误对象
        """
        error_obj = {
            "error": {
                "code": code or 400,
                "message": message,
                "status": status
            }
        }
        if details:
            error_obj["error"]["details"] = details
        return error_obj
    
    @staticmethod
    def create_error(
        message: str,
        status_code: int,
        api_format: ApiFormat = "openai"
    ) -> Dict[str, Any]:
        """
        根据指定格式创建错误响应
        
        Args:
            message: 错误消息
            status_code: HTTP 状态码
            api_format: 目标 API 格式
            
        Returns:
            对应格式的错误对象
        """
        # 根据状态码推断错误类型
        if status_code == 401:
            error_type = "authentication_error"
            gemini_status = "UNAUTHENTICATED"
        elif status_code == 403:
            error_type = "permission_denied_error"
            gemini_status = "PERMISSION_DENIED"
        elif status_code == 404:
            error_type = "not_found_error"
            gemini_status = "NOT_FOUND"
        elif status_code == 429:
            error_type = "rate_limit_error"
            gemini_status = "RESOURCE_EXHAUSTED"
        elif 400 <= status_code < 500:
            error_type = "invalid_request_error"
            gemini_status = "INVALID_ARGUMENT"
        else:
            error_type = "api_error"
            gemini_status = "INTERNAL"
        
        if api_format == "openai":
            return APIError.openai_error(
                message=message,
                error_type=error_type,
                code=str(status_code)
            )
        elif api_format == "gemini":
            return APIError.gemini_error(
                message=message,
                status=gemini_status,
                code=status_code
            )
        else:
            # 默认返回 OpenAI 格式
            return APIError.openai_error(
                message=message,
                error_type=error_type,
                code=str(status_code)
            )


class ErrorConverter:
    """错误格式转换器"""
    
    @staticmethod
    def openai_to_gemini(openai_error: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 OpenAI 格式错误转换为 Gemini 格式
        
        Args:
            openai_error: OpenAI 格式的错误对象
            
        Returns:
            Gemini 格式的错误对象
        """
        error_obj = openai_error.get("error", {})
        message = error_obj.get("message", "未知错误")
        error_type = error_obj.get("type", "api_error")
        code = error_obj.get("code")
        
        # 映射错误类型
        status_map = {
            "authentication_error": "UNAUTHENTICATED",
            "permission_denied_error": "PERMISSION_DENIED",
            "invalid_request_error": "INVALID_ARGUMENT",
            "not_found_error": "NOT_FOUND",
            "rate_limit_error": "RESOURCE_EXHAUSTED",
        }
        
        status = status_map.get(error_type, "UNKNOWN")
        
        try:
            http_code = int(code) if code else 400
        except:
            http_code = 400
        
        return APIError.gemini_error(message, status, http_code)
    
    @staticmethod
    def gemini_to_openai(gemini_error: bytes | Dict[str, Any], status_code: int = None) -> Dict[str, Any]:
        """
        将 Gemini 格式错误转换为 OpenAI 格式
        
        Args:
            gemini_error: Gemini 格式的错误对象（可以是 bytes 或 dict）
            status_code: HTTP 状态码
            
        Returns:
            OpenAI 格式的错误对象
        """
        try:
            if isinstance(gemini_error, bytes):
                gemini_error = json.loads(gemini_error.decode('utf-8'))
            
            error_obj = gemini_error.get("error", {})
            message = error_obj.get("message", "Gemini API 错误")
            status = error_obj.get("status", "UNKNOWN")
            
            # 映射错误类型
            type_map = {
                "INVALID_ARGUMENT": "invalid_request_error",
                "PERMISSION_DENIED": "permission_denied_error",
                "UNAUTHENTICATED": "authentication_error",
                "RESOURCE_EXHAUSTED": "rate_limit_error",
                "NOT_FOUND": "not_found_error",
            }
            
            error_type = type_map.get(status, "api_error")
            
            return APIError.openai_error(
                message=message,
                error_type=error_type,
                code=status
            )
        except Exception as e:
            # 解析失败，返回通用错误
            error_message = f"Gemini Error (HTTP {status_code}): "
            if isinstance(gemini_error, bytes):
                error_message += gemini_error.decode('utf-8', errors='ignore')
            else:
                error_message += str(gemini_error)
            
            return APIError.openai_error(
                message=error_message,
                error_type="api_error",
                code=str(status_code) if status_code else None
            )
    
    @staticmethod
    def convert_upstream_error(
        error_content: bytes | Dict[str, Any],
        status_code: int,
        from_format: ApiFormat,
        to_format: ApiFormat
    ) -> Dict[str, Any]:
        """
        转换上游错误到目标格式
        
        Args:
            error_content: 上游错误内容
            status_code: HTTP 状态码
            from_format: 上游 API 格式
            to_format: 目标格式
            
        Returns:
            目标格式的错误对象
        """
        # 如果格式相同，直接返回（如果是 bytes 需要解析）
        if from_format == to_format:
            if isinstance(error_content, bytes):
                try:
                    return json.loads(error_content.decode('utf-8'))
                except:
                    pass
            return error_content
        
        # 先转换为 OpenAI 格式（中间格式）
        if from_format == "gemini":
            openai_error = ErrorConverter.gemini_to_openai(error_content, status_code)
        else:
            # 已经是 OpenAI 格式或未知格式
            if isinstance(error_content, bytes):
                try:
                    openai_error = json.loads(error_content.decode('utf-8'))
                except:
                    openai_error = APIError.openai_error(
                        message=error_content.decode('utf-8', errors='ignore'),
                        code=str(status_code)
                    )
            else:
                openai_error = error_content
        
        # 再转换为目标格式
        if to_format == "openai":
            return openai_error
        elif to_format == "gemini":
            return ErrorConverter.openai_to_gemini(openai_error)
        else:
            return openai_error


# 便捷函数
def create_api_error(message: str, status_code: int, api_format: ApiFormat = "openai") -> Dict[str, Any]:
    """创建 API 错误响应的便捷函数"""
    return APIError.create_error(message, status_code, api_format)


def convert_error(
    error_content: bytes | Dict[str, Any],
    status_code: int,
    from_format: ApiFormat,
    to_format: ApiFormat
) -> Dict[str, Any]:
    """转换错误格式的便捷函数"""
    return ErrorConverter.convert_upstream_error(error_content, status_code, from_format, to_format)
