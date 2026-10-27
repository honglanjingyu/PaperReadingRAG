"""VISTA 编码服务（必须在 conda 环境 vista 里运行）

协议：JSONL，一行一个请求，一行一个响应。
  请求  {"id": "...", "texts": ["..."], "images": ["/abs/path.png"]}
  响应  {"id": "...", "dim": 1024, "texts": [[...]], "images": [[...]], "error": null}
文本与图片按请求里的顺序各自返回，编码失败的条目返回 null。

启动时向 stderr 打印 "READY <dim>"，主进程据此判断模型已加载完成。
"""

import os
import sys
import json
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PKG = os.getenv("VISTA_PKG_DIR", "/root/visual_bge_pkg")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

WEIGHT = os.getenv("VISTA_WEIGHT", "/root/models/vista/Visualized_m3.pth")
BGE_NAME = os.getenv("VISTA_BGE_NAME", "BAAI/bge-m3")

def log(msg):
    sys.stderr.write("[vista-service] %s\n" % msg)
    sys.stderr.flush()

def main():
    import torch
    from visual_bge.modeling import Visualized_BGE

    log("loading %s from %s" % (BGE_NAME, WEIGHT))
    model = Visualized_BGE(model_name_bge=BGE_NAME, model_weight=WEIGHT)
    model.eval()
    dim = int(model.hidden_dim)
    log("loaded dim=%d device=%s" % (dim, getattr(model, "device", "?")))

    sys.stdout.write("READY %d\n" % dim)
    sys.stdout.flush()

    def encode_batch(key, items, fn):
        out = []
        for it in items or []:
            try:
                if it is None or it == "":
                    out.append(None)
                    continue
                with torch.no_grad():
                    vec = fn(it)
                arr = vec[0].detach().float().cpu().numpy().tolist()
                out.append(arr)
            except Exception as e:
                log("encode %s failed for %r: %s" % (key, str(it)[:80], e))
                out.append(None)
        return out

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "PING":
            sys.stdout.write("PONG\n")
            sys.stdout.flush()
            continue
        req = None
        try:
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                break
            rid = req.get("id")
            texts = encode_batch("text", req.get("texts"), lambda t: model.encode(text=t))
            images = encode_batch("image", req.get("images"), lambda p: model.encode(image=p))
            resp = {"id": rid, "dim": dim, "texts": texts, "images": images, "error": None}
        except Exception as e:
            log("request failed: %s" % traceback.format_exc())
            resp = {"id": (req or {}).get("id"), "dim": dim, "texts": [], "images": [],
                    "error": "%s: %s" % (type(e).__name__, e)}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
