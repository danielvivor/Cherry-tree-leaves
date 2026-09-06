#!/bin/bash

# Create Streamlit config directory
mkdir -p ~/.streamlit/

# Write cloud-friendly server config
cat <<EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
EOF

echo "Streamlit cloud configuration created successfully."