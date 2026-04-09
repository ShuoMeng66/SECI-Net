from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from core.data import default_tokenizer, detokenize_tokens
from core.model import build_model_from_checkpoint

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.pt"
HOST = "127.0.0.1"
PORT = 8000


ASPECT_KEYWORDS = {
    "service": ["客服", "服务", "态度", "回复", "处理", "support", "service", "staff"],
    "logistics": ["物流", "配送", "快递", "发货", "骑手", "delivery", "shipping", "courier"],
    "product": ["产品", "质量", "商品", "功能", "broken", "quality", "feature"],
    "price": ["价格", "退款", "补偿", "收费", "price", "refund", "cost"],
    "experience": ["体验", "系统", "页面", "卡顿", "闪退", "experience", "app", "website"],
    "environment": ["环境", "卫生", "包装", "噪音", "房间", "clean", "noise", "package"],
}

ASPECT_LABELS = {
    "service": "Service",
    "logistics": "Logistics",
    "product": "Product",
    "price": "Price",
    "experience": "Experience",
    "environment": "Environment",
}


def count_hits(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?\n\r]+", text)
    return [part.strip() for part in parts if part.strip()]


def build_aspect_scores(text: str) -> dict[str, int]:
    aspect_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = count_hits(text, keywords)
        aspect_scores[aspect] = min(100, 18 + hits * 16)
    return aspect_scores


def build_actions(primary_aspect: str, recoverability: int, severity: int) -> list[dict[str, str]]:
    aspect_action = {
        "service": ("优先处理客服响应", "尽快给出明确回复与处理时点。"),
        "logistics": ("优先处理履约问题", "核查配送节点并同步补救方案。"),
        "product": ("优先处理产品问题", "确认质量或功能异常并提供售后方案。"),
        "price": ("优先处理价格争议", "核对退款、补偿或价格说明。"),
        "experience": ("优先处理体验问题", "检查流程、页面或系统异常。"),
        "environment": ("优先处理环境问题", "核查环境、包装或现场条件。"),
    }
    title, body = aspect_action.get(primary_aspect, ("优先处理当前问题", "先核查事实，再给出动作。"))
    actions = [{"title": title, "body": body}]
    if severity >= 70:
        actions.append({"title": "升级处理", "body": "将当前样本升级给更高优先级负责人。"})
    if recoverability >= 65:
        actions.append({"title": "尝试服务恢复", "body": "优先围绕主维度给出补救动作，争取态度回转。"})
    return actions


def fallback_evidence(text: str, sentiment_score: int, primary_aspect: str, aspect_scores: dict[str, int]) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    snippets = sentences[:3] or [text.strip() or text]
    return [
        {
            "text": snippet,
            "sentiment": sentiment_score,
            "aspects": [
                {
                    "key": primary_aspect,
                    "label": ASPECT_LABELS[primary_aspect],
                    "score": aspect_scores[primary_aspect],
                }
            ],
        }
        for snippet in snippets
    ]


class ModelRuntime:
    def __init__(self, checkpoint_path: str | Path = MODEL_PATH, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model, payload = build_model_from_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.vocab = self._load_vocab(payload["vocab"])
        self.labels = payload["label_to_index"]
        self.model_name = self.checkpoint_path.name
        self.max_len = int(payload["model_config"].get("max_len", 512))
        self.block_size = int(payload["model_config"].get("block_size", 16))
        self.positive_class_index = int(
            payload["model_config"].get("positive_class_index", max(self.labels.values()))
        )

    def _load_vocab(self, payload: dict[str, Any]) -> dict[str, Any]:
        itos = list(payload["itos"])
        return {
            **payload,
            "itos": itos,
            "stoi": {token: index for index, token in enumerate(itos)},
        }

    def encode_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = default_tokenizer(text)[: self.max_len]
        token_ids = [self.vocab["stoi"].get(token, self.vocab["stoi"].get("<unk>", 1)) for token in tokens]
        if not token_ids:
            token_ids = [self.vocab["stoi"].get("<unk>", 1)]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)
        return input_ids, attention_mask, lengths

    def build_evidence(
        self,
        text: str,
        evidence_indices: torch.Tensor,
        evidence_scores: torch.Tensor,
        sentiment_score: int,
        primary_aspect: str,
        aspect_scores: dict[str, int],
    ) -> list[dict[str, Any]]:
        tokens = default_tokenizer(text)[: self.max_len]
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for block_index, block_score in zip(evidence_indices.tolist(), evidence_scores.tolist()):
            if block_score <= 0:
                continue
            start = block_index * self.block_size
            if start >= len(tokens):
                continue
            end = min(start + self.block_size, len(tokens))
            snippet = detokenize_tokens(tokens[start:end]).strip()
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            items.append(
                {
                    "text": snippet,
                    "sentiment": sentiment_score,
                    "aspects": [
                        {
                            "key": primary_aspect,
                            "label": ASPECT_LABELS[primary_aspect],
                            "score": max(aspect_scores[primary_aspect], int(round(block_score * 100))),
                        }
                    ],
                }
            )
        return items

    @torch.inference_mode()
    def predict(self, text: str) -> dict[str, Any]:
        input_ids, attention_mask, lengths = self.encode_text(text)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            return_dict=True,
        )
        probabilities = torch.softmax(output.logits, dim=-1).squeeze(0).cpu()
        positive_prob = float(probabilities[self.positive_class_index].item())
        if probabilities.numel() == 2:
            negative_prob = float(probabilities[1 - self.positive_class_index].item())
        else:
            negative_prob = float(max(0.0, 1.0 - positive_prob))

        aspect_scores = build_aspect_scores(text)
        primary_aspect = max(aspect_scores, key=aspect_scores.get)
        sentiment_score = int(round((positive_prob - negative_prob) * 100))
        evidence = self.build_evidence(
            text=text,
            evidence_indices=output.evidence_indices.squeeze(0).cpu(),
            evidence_scores=output.evidence_scores.squeeze(0).cpu(),
            sentiment_score=sentiment_score,
            primary_aspect=primary_aspect,
            aspect_scores=aspect_scores,
        )
        if not evidence:
            evidence = fallback_evidence(
                text=text,
                sentiment_score=sentiment_score,
                primary_aspect=primary_aspect,
                aspect_scores=aspect_scores,
            )

        recoverability = max(0, min(100, int(round(float(output.recoverability.item()) * 100))))
        confidence = max(0, min(100, int(round(float(probabilities.max().item()) * 100))))
        evidence_strength = int(round(float(output.evidence_scores.mean().item()) * 100))
        severity = min(
            100,
            int(round(30 + negative_prob * 55 + aspect_scores[primary_aspect] * 0.08 + evidence_strength * 0.08)),
        )
        return {
            "originalText": text,
            "sentimentScore": sentiment_score,
            "severity": severity,
            "recoverability": recoverability,
            "confidence": confidence,
            "primaryAspectKey": primary_aspect,
            "aspectScores": aspect_scores,
            "evidence": evidence,
            "actions": build_actions(primary_aspect, recoverability=recoverability, severity=severity),
            "tags": [],
            "counterfactual": [],
            "sentenceTrend": [item["sentiment"] for item in evidence],
            "model": self.model_name,
            "probabilities": {
                "positive": round(positive_prob, 6),
                "negative": round(negative_prob, 6),
            },
        }


_RUNTIME: ModelRuntime | None = None
_RUNTIME_ERROR: str | None = None


def get_runtime() -> ModelRuntime:
    global _RUNTIME, _RUNTIME_ERROR
    if _RUNTIME is None and _RUNTIME_ERROR is None:
        try:
            _RUNTIME = ModelRuntime()
        except Exception as exc:
            _RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
    if _RUNTIME is None:
        raise RuntimeError(_RUNTIME_ERROR or "Model runtime is unavailable.")
    return _RUNTIME


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            try:
                runtime = get_runtime()
                self._write_json({"ok": True, "model": runtime.model_name})
            except RuntimeError as exc:
                self._write_json({"ok": False, "error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(raw_body.decode("utf-8") or "{}")
        text = str(payload.get("text", "")).strip()
        if not text:
            self._write_json({"error": "text is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            analysis = get_runtime().predict(text)
        except RuntimeError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._write_json({"analysis": analysis, "mode": "remote"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"API listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
