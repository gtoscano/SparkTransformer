# QA Chat - Cloudflare Workers AI

A chat application deployed entirely on Cloudflare using Workers AI. No API keys needed — the Worker has direct access to AI models via the `AI` binding.

## Prerequisites

- **Node.js** v18 or higher — [download here](https://nodejs.org/)
- **A Cloudflare account** — sign up for free at https://dash.cloudflare.com/sign-up

## Step-by-Step Deployment

### Step 1: Install Dependencies

```bash
cd qa_cloudflare_worker
npm install
```

This installs `wrangler`, the Cloudflare CLI tool.

### Step 2: Log In to Cloudflare

```bash
npx wrangler login
```

This opens your browser. Log in to your Cloudflare account and authorize Wrangler. You only need to do this once.

To verify you're logged in:

```bash
npx wrangler whoami
```

### Step 3: Test Locally

```bash
npm run dev
```

Open http://localhost:8787 in your browser. Type a message and verify the chat works.

Press `Ctrl+C` to stop the local server.

### Step 4: Deploy to Cloudflare

```bash
npm run deploy
```

On first deploy, Wrangler may ask you to set up a `workers.dev` subdomain. Follow the prompts.

When complete, you'll see output like:

```
Published qa-chat (0.5 sec)
  https://qa-chat.your-subdomain.workers.dev
```

That URL is your live, publicly accessible chat app.

### Step 5: Verify

Open the URL from Step 4 in your browser and send a test message.

## Updating the App

After making changes to `src/worker.js`, redeploy with:

```bash
npm run deploy
```

## Custom Domain (Optional)

To use your own domain instead of `*.workers.dev`:

1. Go to https://dash.cloudflare.com > **Workers & Pages**
2. Click on your `qa-chat` worker
3. Go to **Settings** > **Triggers** > **Custom Domains**
4. Add your domain (it must be on Cloudflare DNS)

## Project Structure

```
qa_cloudflare_worker/
├── src/
│   └── worker.js       # Worker code: serves HTML + handles AI API calls
├── wrangler.toml        # Cloudflare Worker configuration
├── package.json         # Node.js dependencies and scripts
└── README.md
```

## How It Works

- **`/`** — serves the HTML/JS chat interface
- **`/api/chat`** — backend endpoint that calls Cloudflare Workers AI
- The `[ai]` binding in `wrangler.toml` gives the Worker direct access to AI models — no API tokens or keys needed at runtime
- Everything runs on Cloudflare's edge network (300+ locations worldwide)

## Available Models

| Model | Size | Notes |
|-------|------|-------|
| Llama 3.1 8B Instruct | 8B | Best overall quality (default) |
| Mistral 7B Instruct | 7B | Fast, good for general tasks |
| Llama 2 7B Chat | 7B | Older but reliable |
| Zephyr 7B Beta (AWQ) | 7B | Quantized, fast inference |
| Falcon 7B Instruct | 7B | Good for instruction following |

Full model catalog: https://developers.cloudflare.com/workers-ai/models/

## Pricing

Cloudflare Workers AI has a free tier:

- **Workers**: 100,000 requests/day free
- **Workers AI**: 10,000 neurons/day free (enough for ~100-300 chat messages depending on length)

See https://developers.cloudflare.com/workers-ai/platform/pricing/ for details.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `wrangler: command not found` | Run `npm install` first |
| `Authentication error` | Run `npx wrangler login` again |
| `AI binding not found` | Make sure `wrangler.toml` has the `[ai]` section |
| Model returns empty response | Try a different model from the dropdown |
| 429 Too Many Requests | You've hit the free tier limit — wait until the next day |
