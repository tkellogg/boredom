from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


# Lightweight interface so plugins can be developed independently of internal types
@dataclass
class PluginContext:
    config: Any
    state: Any
    tool_registry: Any
    _append_message: Callable[[Dict[str, Any]], None]

    def emit_system(self, text: str) -> None:
        """Append a visible system message (not sent to the model input payload).

        Note: Our loop only sends the current user tick and tool messages; these
        system entries are for the log/HTML and are not injected into the
        provider request stream.
        """
        msg = {"role": "system", "content": [{"type": "output_text", "text": str(text)}]}
        self._append_message(msg)

    def emit_note(self, text: str, *, tag: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> None:
        """Append a plugin note invisible to the AI but visible in the HTML.

        The entry is recorded with role='note' so renderers can style it.
        """
        msg: Dict[str, Any] = {
            "role": "note",
            "content": [{"type": "output_text", "text": str(text)}],
        }
        if tag:
            msg["note_tag"] = str(tag)
        if data is not None:
            msg["note_data"] = data
        self._append_message(msg)


class BasePlugin:
    """Base class for all plugins.

    Override the hooks you need; default behavior is a no-op pass-through.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    # Lifecycle
    def on_attach(self, ctx: PluginContext) -> None:  # called once before the loop starts
        self.ctx = ctx

    # Messages
    def transform_system_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return message

    def transform_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        return message

    # Tools
    def transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a transformed list of tool specs (add/remove/modify).

        This replaces the old 'filter_tools' hook.
        """
        return tools

    # Back-compat shim (deprecated)
    def filter_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # pragma: no cover
        return self.transform_tools(tools)

    # Requests / responses
    def before_request(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return request_kwargs

    def after_response(self, response_dict: Dict[str, Any]) -> None:
        pass

    def after_tool_result(self, tool_log: Dict[str, Any]) -> None:
        pass

    def on_iteration_end(self) -> None:
        pass


@dataclass
class PluginSpec:
    module: str
    cls_name: str = "Plugin"
    params: Dict[str, Any] = field(default_factory=dict)


class PluginManager:
    def __init__(self, plugin_dir: Path, specs: Iterable[PluginSpec]):
        self.plugin_dir = Path(plugin_dir)
        self.specs = list(specs)
        self.plugins: List[BasePlugin] = []

    def load(self) -> None:
        # Ensure the parent of the plugin directory is importable so we can load
        # modules as `plugins.module`, allowing relative imports inside plugins.
        parent = str(self.plugin_dir.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        self.plugins = []
        package = self.plugin_dir.name  # typically 'plugins'
        for spec in self.specs:
            module_name = str(spec.module)
            candidates = []
            if module_name.startswith(package + "."):
                candidates = [module_name]
            elif "." in module_name and module_name.split(".", 1)[0] == package:
                candidates = [module_name]
            else:
                # Prefer package-qualified, then raw fallback
                candidates = [f"{package}.{module_name}", module_name]
            last_exc: Optional[Exception] = None
            mod = None
            for mn in candidates:
                try:
                    mod = importlib.import_module(mn)
                    module_name = mn
                    break
                except Exception as e:  # try next candidate
                    last_exc = e
                    continue
            if mod is None:
                raise last_exc or ImportError(f"Could not import plugin module {spec.module}")
            cls = getattr(mod, spec.cls_name)
            inst = cls(**(spec.params or {}))
            # helpful for logging
            try:
                inst.name = getattr(inst, "name", module_name) or module_name
            except Exception:
                pass
            self.plugins.append(inst)

    def attach(self, config: Any, state: Any, tool_registry: Any, append_message: Callable[[Dict[str, Any]], None]) -> None:
        ctx = PluginContext(config=config, state=state, tool_registry=tool_registry, _append_message=append_message)
        for p in self.plugins:
            p.on_attach(ctx)

    # Fan-out helpers
    def transform_system_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        for p in self.plugins:
            message = p.transform_system_message(message)
        return message

    def transform_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        for p in self.plugins:
            message = p.transform_user_message(message)
        return message

    def transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for p in self.plugins:
            # Prefer new hook; fall back to legacy if needed
            if hasattr(p, "transform_tools"):
                tools = p.transform_tools(tools)  # type: ignore[attr-defined]
            else:  # pragma: no cover
                tools = p.filter_tools(tools)  # type: ignore[attr-defined]
        return tools

    # Back-compat for callers (deprecated)
    def filter_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # pragma: no cover
        return self.transform_tools(tools)

    def before_request(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        for p in self.plugins:
            request_kwargs = p.before_request(request_kwargs)
        return request_kwargs

    def after_response(self, response_dict: Dict[str, Any]) -> None:
        for p in self.plugins:
            p.after_response(response_dict)

    def after_tool_result(self, tool_log: Dict[str, Any]) -> None:
        for p in self.plugins:
            p.after_tool_result(tool_log)

    def on_iteration_end(self) -> None:
        for p in self.plugins:
            p.on_iteration_end()

    @staticmethod
    def parse_specs_from_json(raw: str) -> List[PluginSpec]:
        try:
            data = json.loads(raw) if raw else None
        except Exception as e:
            raise ValueError(f"Invalid --plugins JSON: {e}")
        if not data:
            return []
        if not isinstance(data, list):
            raise ValueError("--plugins must be a JSON list")
        specs: List[PluginSpec] = []
        for item in data:
            if not isinstance(item, dict) or not item.get("module"):
                raise ValueError("Each plugin entry must be an object with a 'module' field")
            specs.append(PluginSpec(module=str(item["module"]), cls_name=str(item.get("class", "Plugin")), params=item.get("params") or {}))
        return specs
