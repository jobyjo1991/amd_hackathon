import json
from openai import OpenAI
from transformers import AutoTokenizer

class LocalComplianceEngine:
    def __init__(self, base_url="http://localhost:8000/v1", model_id="Qwen/Qwen2.5-14B-Instruct"):
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key="local-vllm")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def _hf_token_chunker(self, text: str, max_tokens=1500, overlap_tokens=150) -> list:
        all_tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        total_len = len(all_tokens)
        
        while start < total_len:
            end = min(start + max_tokens, total_len)
            chunks.append(self.tokenizer.decode(all_tokens[start:end], skip_special_tokens=True))
            if end == total_len:
                break
            start = end - overlap_tokens
        return chunks

    def compare_documents(self, doc_a_text: str, doc_b_text: str, rule: dict) -> dict:
        """
        Compares Doc A and Doc B segmentally against a specific compliance target rule.
        """
        chunks_a = self._hf_token_chunker(doc_a_text)
        chunks_b = self._hf_token_chunker(doc_b_text)
        
        context_a = chunks_a[0] if chunks_a else "No text found in Document A"
        context_b = chunks_b[0] if chunks_b else "No text found in Document B"

        prompt = f"""
        You are a senior automated text analytics system verifying version drift.
        Compare the text provided in Document A against Document B to check for structural changes.
        
        [EVALUATION RULE FOCUS SCOPE]
        ID: {rule['id']}
        Scope Area: {rule['text']}
        
        [DOCUMENT A CONTEXT (Baseline Source)]
        {context_a[:4000]}
        
        [DOCUMENT B CONTEXT (Target Version)]
        {context_b[:4000]}
        
        [TASK]
        1. Ignore whether the text passes external regulations. Focus ONLY on whether Document B's text has drifted or changed compared to Document A.
        2. If the text blocks contain the exact same wording, metrics, or terms regarding this scope area, you MUST return "MATCH".
        3. If phrases, timelines, values, or clauses have changed, return "MISMATCH".
        4. If the clause is present in Doc A but completely deleted or missing from Doc B, return "GAP".
        
        Respond strictly with a single JSON block:
        {{
            "status": "MATCH" or "MISMATCH" or "GAP",
            "confidence": 1.0,
            "doc_a_evidence": "Verbatim quote from Doc A",
            "doc_b_evidence": "Verbatim quote from Doc B",
            "explanation": "Provide a literal translation delta comparing the text strings."
        }}
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_content = res.choices[0].message.content.strip()
            clean_json = raw_content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            return {
                "status": "GAP",
                "confidence": 0.0,
                "doc_a_evidence": "N/A",
                "doc_b_evidence": "N/A",
                "explanation": f"Failsafe triggered. Comparison processing error: {str(e)}"
            }