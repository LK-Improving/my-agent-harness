"""本机验收代理：首次写请求注入写后响应失败，其余请求原样转发。"""

import httpx
from fastapi import FastAPI, Request, Response

app = FastAPI()
injected = False


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request) -> Response:
    global injected
    headers = {
        "Authorization": request.headers.get("authorization", ""),
        "Content-Type": "application/json",
    }
    if request.method == "POST" and path.startswith("writes/") and not injected:
        injected = True
        headers["X-Sandbox-Fault"] = "after_write"
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        result = await client.request(
            request.method,
            "http://127.0.0.1:18081/" + path,
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
    print("验收代理", request.method, path.split("/")[0], result.status_code, flush=True)
    return Response(result.content, status_code=result.status_code, media_type="application/json")
