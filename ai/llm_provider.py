"""LLM Provider Abstraction — vendor-neutral."""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
class LLMProvider(ABC):
    name="base"
    @abstractmethod
    def available(self)->bool: ...
    @abstractmethod
    def complete(self, system, user, max_tokens=700)->str: ...
class AnthropicProvider(LLMProvider):
    name="anthropic"
    def available(self): return bool(os.environ.get("ANTHROPIC_API_KEY"))
    def complete(self, system, user, max_tokens=700):
        import anthropic
        c=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        m=c.messages.create(model=os.environ.get("ACI_MODEL","claude-sonnet-4-5"),max_tokens=max_tokens,system=system,messages=[{"role":"user","content":user}])
        return "".join(getattr(b,"text","") for b in m.content).strip()
class AzureOpenAIProvider(LLMProvider):
    name="azure-openai"
    def available(self): return bool(os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"))
    def complete(self, system, user, max_tokens=700):
        from openai import AzureOpenAI
        c=AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],api_version=os.environ.get("AZURE_OPENAI_API_VERSION","2024-02-15-preview"))
        r=c.chat.completions.create(model=os.environ.get("AZURE_OPENAI_DEPLOYMENT","gpt-4o"),max_tokens=max_tokens,messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
class OpenAIProvider(LLMProvider):
    name="openai"
    def available(self): return bool(os.environ.get("OPENAI_API_KEY"))
    def complete(self, system, user, max_tokens=700):
        from openai import OpenAI
        c=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        r=c.chat.completions.create(model=os.environ.get("ACI_MODEL","gpt-4o"),max_tokens=max_tokens,messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
def get_provider():
    pref=os.environ.get("ACI_LLM","").lower()
    provs={p.name:p for p in [AnthropicProvider(),AzureOpenAIProvider(),OpenAIProvider()]}
    if pref in provs and provs[pref].available(): return provs[pref]
    for p in provs.values():
        if p.available(): return p
    return AnthropicProvider()
