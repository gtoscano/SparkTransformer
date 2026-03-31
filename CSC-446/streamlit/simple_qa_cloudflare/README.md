# QA Chat with Cloudflare Workers AI

A simple chat application using Cloudflare Workers AI and Streamlit.

## Setup

### 1. Get Cloudflare Credentials

You need two values from your Cloudflare account:

#### Account ID
1. Log in to https://dash.cloudflare.com
2. Select your account
3. Go to **Workers & Pages** in the left sidebar
4. Your **Account ID** is displayed on the right side of the overview page

#### API Token
1. Go to https://dash.cloudflare.com/profile/api-tokens
2. Click **Create Token**
3. Use the **Custom token** template
4. Set permissions:
   - **Account** > **Workers AI** > **Read**
5. Click **Continue to summary** > **Create Token**
6. Copy the token (you won't be able to see it again)

### 2. Install Dependencies

```bash
pip install streamlit requests
```

### 3. Run the App

```bash
streamlit run app.py
```

Then enter your **Account ID** and **API Token** in the sidebar.

### Optional: Use Environment Variables

You can also set credentials as environment variables to avoid entering them each time:

```bash
export CLOUDFLARE_ACCOUNT_ID="your_account_id"
export CLOUDFLARE_API_TOKEN="your_api_token"
```

## Available Models

| Model | Size | Notes |
|-------|------|-------|
| `@cf/meta/llama-3.1-8b-instruct` | 8B | Best overall quality (default) |
| `@cf/mistralai/mistral-7b-instruct-v0.1` | 7B | Fast, good for general tasks |
| `@cf/meta/llama-2-7b-chat-fp16` | 7B | Older but reliable |
| `@cf/thebloke/zephyr-7b-beta-awq` | 7B | Quantized, fast inference |
| `@cf/tiiuae/falcon-7b-instruct` | 7B | Good for instruction following |

Full model catalog: https://developers.cloudflare.com/workers-ai/models/
