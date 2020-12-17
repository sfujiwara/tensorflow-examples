curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d @data/example01.json \
  http://localhost:8501/v1/models/mnist:predict
