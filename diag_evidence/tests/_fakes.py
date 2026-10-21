"""测试替身：返回与 tools._common.http_call 完全一致的 (kind, body, error) 三元组。

注意：被测工具通过依赖注入接收 transport，因此测试不需要 patch requests，
也不依赖任何真实服务（无卡环境可跑）。
"""

from __future__ import annotations


def ok(body):
    def transport(method, url, **kwargs):
        return "ok", body, ""
    return transport


def client_error(body, message="bad request"):
    def transport(method, url, **kwargs):
        return "client", body, message
    return transport


def server_error(body=None, message="internal error"):
    def transport(method, url, **kwargs):
        return "server", body or {"error": message}, message
    return transport


def not_found(message="not found"):
    def transport(method, url, **kwargs):
        return "not_found", None, message
    return transport


def connection_error(message="连接被拒绝"):
    def transport(method, url, **kwargs):
        return "conn", None, message
    return transport


def routed(rules):
    """按 URL 子串分派的替身。

    rules: {url_substring: transport}
    """
    def transport(method, url, **kwargs):
        for key, handler in rules.items():
            if key in url:
                return handler(method, url, **kwargs)
        raise AssertionError(f"未预期的 URL: {url}")
    return transport
