#!/bin/bash

# 1. تأكد من وجود cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "🚧 cloudflared غير موجود، يتم تثبيته..."
    sudo apt update && sudo apt install cloudflared -y
fi

# 2. شغّل cloudflared في الخلفية وسجّل الإخراج في ملف
echo "🚀 يتم تشغيل النفق الآن ..."
cloudflared tunnel --url http://localhost:3000 > tunnel.log 2>&1 &

# 3. انتظر حتى يظهر الرابط
echo "⏳ بانتظار الرابط ..."
while ! grep -q "trycloudflare.com" tunnel.log; do
  sleep 1
done

# 4. استخراج الرابط وعرضه
LINK=$(grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' tunnel.log | head -n1)
echo "✅ رابطك: $LINK"
