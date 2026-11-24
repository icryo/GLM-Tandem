# Stand-in for the model server: takes 8 s to start, answers /health, /v1/models and chat completions.
import asyncio, json, sys, time
from aiohttp import web
async def health(r): return web.json_response({"status": "ok"})
async def models(r): return web.json_response({"data": [{"id": "default_model"}]})
async def chat(r):
    b = await r.json(); t = float(b.get("fake_seconds", 2))
    if b.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await resp.prepare(r)
        for i in range(5):
            await asyncio.sleep(t / 5)
            await resp.write(("data: " + json.dumps({"choices": [{"delta": {"content": f"w{i} "}}]}) + "\n\n").encode())
        await resp.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}) + "\n\ndata: [DONE]\n\n").encode())
        await resp.write_eof(); return resp
    await asyncio.sleep(t)
    return web.json_response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
time.sleep(8)
app = web.Application(); app.router.add_get("/health", health); app.router.add_get("/v1/models", models); app.router.add_post("/v1/chat/completions", chat)
web.run_app(app, host="127.0.0.1", port=8011, print=lambda *_: None)
