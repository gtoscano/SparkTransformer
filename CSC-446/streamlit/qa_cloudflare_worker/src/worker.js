export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Serve the frontend
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(HTML, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    // API endpoint for chat
    if (url.pathname === "/api/chat" && request.method === "POST") {
      const { messages, model, max_tokens, temperature } = await request.json();

      const response = await env.AI.run(model || "@cf/meta/llama-3.1-8b-instruct", {
        messages,
        max_tokens: max_tokens || 256,
        temperature: temperature || 0.7,
      });

      return Response.json({ response: response.response });
    }

    return new Response("Not found", { status: 404 });
  },
};

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QA Chat - Cloudflare Workers AI</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f5f5;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #f48120;
      color: white;
      padding: 16px 24px;
      font-size: 1.2em;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .settings {
      background: white;
      padding: 12px 24px;
      border-bottom: 1px solid #ddd;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 0.85em;
    }
    .settings label { font-weight: 500; }
    .settings select, .settings input {
      padding: 4px 8px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 0.9em;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .message {
      max-width: 75%;
      padding: 12px 16px;
      border-radius: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .user {
      align-self: flex-end;
      background: #f48120;
      color: white;
      border-bottom-right-radius: 4px;
    }
    .assistant {
      align-self: flex-start;
      background: white;
      border: 1px solid #ddd;
      border-bottom-left-radius: 4px;
    }
    .typing {
      align-self: flex-start;
      background: white;
      border: 1px solid #ddd;
      border-bottom-left-radius: 4px;
      color: #888;
      font-style: italic;
    }
    #input-area {
      padding: 16px 24px;
      background: white;
      border-top: 1px solid #ddd;
      display: flex;
      gap: 12px;
    }
    #user-input {
      flex: 1;
      padding: 12px 16px;
      border: 1px solid #ccc;
      border-radius: 8px;
      font-size: 1em;
      outline: none;
    }
    #user-input:focus { border-color: #f48120; }
    #send-btn {
      padding: 12px 24px;
      background: #f48120;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1em;
      cursor: pointer;
      font-weight: 500;
    }
    #send-btn:hover { background: #d9711a; }
    #send-btn:disabled { background: #ccc; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>&#9729;&#65039; QA Chat &mdash; Cloudflare Workers AI</header>

  <div class="settings">
    <label>Model:
      <select id="model">
        <option value="@cf/meta/llama-3.1-8b-instruct">Llama 3.1 8B</option>
        <option value="@cf/mistralai/mistral-7b-instruct-v0.1">Mistral 7B</option>
        <option value="@cf/meta/llama-2-7b-chat-fp16">Llama 2 7B</option>
        <option value="@cf/thebloke/zephyr-7b-beta-awq">Zephyr 7B</option>
        <option value="@cf/tiiuae/falcon-7b-instruct">Falcon 7B</option>
      </select>
    </label>
    <label>Max tokens: <input type="number" id="max-tokens" value="256" min="64" max="1024" step="64"></label>
    <label>Temperature: <input type="number" id="temperature" value="0.7" min="0.1" max="1.5" step="0.1"></label>
  </div>

  <div id="chat-container"></div>

  <div id="input-area">
    <input type="text" id="user-input" placeholder="Ask me anything..." autocomplete="off">
    <button id="send-btn">Send</button>
  </div>

  <script>
    const chatContainer = document.getElementById("chat-container");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const messages = [];
    const systemPrompt = "You are a helpful assistant. Answer questions clearly and concisely.";

    function addMessage(role, content) {
      const div = document.createElement("div");
      div.className = "message " + role;
      div.textContent = content;
      chatContainer.appendChild(div);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return div;
    }

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;

      userInput.value = "";
      sendBtn.disabled = true;

      addMessage("user", text);
      messages.push({ role: "user", content: text });

      const typingDiv = addMessage("typing", "Thinking...");

      const apiMessages = [
        { role: "system", content: systemPrompt },
        ...messages,
      ];

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: apiMessages,
            model: document.getElementById("model").value,
            max_tokens: parseInt(document.getElementById("max-tokens").value),
            temperature: parseFloat(document.getElementById("temperature").value),
          }),
        });

        const data = await res.json();
        typingDiv.remove();

        const reply = data.response || "No response received.";
        addMessage("assistant", reply);
        messages.push({ role: "assistant", content: reply });
      } catch (err) {
        typingDiv.remove();
        addMessage("assistant", "Error: " + err.message);
      }

      sendBtn.disabled = false;
      userInput.focus();
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendMessage();
    });

    userInput.focus();
  </script>
</body>
</html>`;
