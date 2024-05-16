from pydantic import BaseModel

class GetResponseParams(BaseModel):
    query_content: str
    app_name: str