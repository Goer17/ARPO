import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GoogleSearchTool(BaseTool):
    """Google search tool adapter for sglang multi-turn function calling."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.domain = config.get("domain")
        if not self.domain:
            raise ValueError("GoogleSearchTool config must include 'domain'")

        self.job_name = config.get("job_name", "arpo_deepsearch_exp")
        self.backend = config.get("backend", "aib")
        self.timeout_ms = int(config.get("timeout_ms", 200000))

        # Lazy import keeps module import safe when this tool is not used.
        from coral.client.rtp_client import RtpClient

        self.client = RtpClient(self.domain)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"history": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        query_list = parameters.get("query_list", None)
        if isinstance(query_list, list):
            queries = [str(q) for q in query_list if str(q).strip()]
        elif "query" in parameters:
            queries = [str(parameters["query"]).strip()]
        else:
            queries = []

        if not queries:
            msg = json.dumps({"error": "query_list is required and cannot be empty"}, ensure_ascii=False)
            return msg, 0.0, {"status": "invalid_parameters", "query_count": 0}

        request_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_000")
        payload = {
            "job_name": self.job_name,
            "enable_cache": True,
            "api_name": "google_search",
            "backend": self.backend,
            "request_id": request_id,
            "querys": queries,
        }

        try:
            response = self.client.process(payload, path="/", timeout_ms=self.timeout_ms)
            result = response["result"] if isinstance(response, dict) and "result" in response else response
            result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
            self._instance_dict[instance_id]["history"].append(result_text)
            return result_text, 0.0, {"status": "ok", "query_count": len(queries)}
        except Exception as e:
            logger.exception("GoogleSearchTool execute failed")
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            return err, 0.0, {"status": "error", "query_count": len(queries)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
