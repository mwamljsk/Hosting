[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:3000
Restart=on-failure
RestartSec=5s
User=root

[Install]
WantedBy=multi-user.target
