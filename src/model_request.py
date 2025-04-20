from ollama import chat


class LLM:
    
    def __init__(self, model="deepseek-r1:1.5b", chat_ctx=[]):
        self._model = model
        self._chat_ctx = chat_ctx
    

    def request(self, message, stream=False):
        if message:
            response = chat(model=self._model, 
                            messages=self._chat_ctx + [{ 'role': 'user', 'content': message }], 
                            stream=stream)
            
            self._chat_ctx += [
              { 'role': 'user', 'content': message },
              { 'role': 'assistant', 'content': response.message.content }
            ]

            return ' '.join(response.message.content.split("</think>")[1].split())
        
        return None


