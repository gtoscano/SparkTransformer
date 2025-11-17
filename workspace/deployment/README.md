docker compose up -d vllm-qlora


#  Testing the qlora model
curl http://localhost:8020/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "imdb",
        "messages": [
          {
            "role": "user",
            "content": "Write a very positive and enthusiastic review of the movie The Matrix."
          }
        ],
        "max_tokens": 200,
        "temperature": 0.8
      }'


curl http://localhost:8020/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "imdb",
        "messages": [
          {"role": "user", "content": "Write a positive review of The Matrix."}
        ]
      }'

curl http://localhost:8020/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "imdb",
        "messages": [
          {"role": "user", "content": "Was The Matrix a good movie?"}
        ]
      }'


# Testing the original llama:
curl http://localhost:8020/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
          {"role": "user", "content": "Was The Matrix a good movie?"}
        ]
      }'

