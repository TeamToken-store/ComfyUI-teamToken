import { app } from "../../scripts/app.js";

// Adds a "teamToken" pane to ComfyUI Settings. The values are persisted by
// ComfyUI into user/<user>/comfy.settings.json, which the Python side reads
// (teamtoken/settings.py). The per-node api_key input still overrides these,
// so users on ComfyUI builds without this settings API are never stuck.
app.registerExtension({
  name: "teamToken.settings",
  settings: [
    {
      id: "teamToken.apiKey",
      name: "teamToken API key",
      category: ["teamToken", "Credentials", "API key"],
      type: "text",
      defaultValue: "",
      tooltip:
        "Your sk-… key from app.teamtoken.store → Keys. Used by all teamToken nodes " +
        "unless a node's api_key input is set.",
      attrs: { type: "password", autocomplete: "off" },
    },
    {
      id: "teamToken.serverUrl",
      name: "teamToken server URL",
      category: ["teamToken", "Credentials", "Server URL"],
      type: "text",
      defaultValue: "https://api.teamtoken.store",
      tooltip: "Gateway base URL. Leave as default unless you run a private gateway.",
    },
  ],
});
