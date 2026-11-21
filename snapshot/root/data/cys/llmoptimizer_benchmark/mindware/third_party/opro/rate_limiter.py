import time
import tiktoken
import asyncio

class RateLimiter:
    def __init__(self, max_tokens, time_frame, max_requests=700):
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.time_frame = time_frame
        self.timestamps = []
        self.tokens_used = []
        self.request_count = 0

    def _prune(self, current_time):
        """移除时间窗口之外的旧记录"""
        while self.timestamps and self.timestamps[0] < current_time - self.time_frame:
            self.timestamps.pop(0)
            self.tokens_used.pop(0)
            # 注意：在修剪时，我们不减少 request_count，因为它只关心窗口内的添加次数
            # 正确的逻辑是在添加新请求时再修剪和计算
    
    async def check_and_wait(self, request_text: str, current_time: float):
        """在发送请求前检查并等待，这是一个异步方法"""
        self._prune(current_time)

        # 检查请求频率限制
        if len(self.timestamps) >= self.max_requests:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            if sleep_time > 0:
                print(f'[Rate Limiter] Request limit reached. Sleeping for {sleep_time:.2f}s...')
                await asyncio.sleep(sleep_time)
        
        # 预估本次请求的 token 并检查 token 限制
        # 注意：这里只估算输入，因为我们还不知道输出
        try:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
            num_tokens = len(encoding.encode(request_text))
        except Exception:
            num_tokens = int(len(request_text) / 3) # 简化的回退估算

        if sum(self.tokens_used) + num_tokens > self.max_tokens:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            if sleep_time > 0:
                print(f'[Rate Limiter] Token limit likely to be exceeded. Sleeping for {sleep_time:.2f}s...')
                await asyncio.sleep(sleep_time)

    def add_request_info(self, total_tokens: int, current_time: float):
        """在收到响应后，用准确的 token 数量更新记录"""
        self._prune(current_time)
        self.timestamps.append(current_time)
        self.tokens_used.append(total_tokens)