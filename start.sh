#!/bin/bash

# تأكد من تثبيت cloudflared
if ! command -v cloudflared &> /dev/null
then
    echo "يتم تثبيت cloudflared ..."
    sudo apt update && sudo apt install cloudflared -y
fi

# تشغيل cloudflared بنفق ل localhost:3000 وحفظ اللوق
cloudflared tunnel --url http://localhost:3000 > cloudflare.log 2>&1 &

# حفظ PID للعملية
PID=$!

echo "انتظار ظهور الرابط في السجلات..."

# ننتظر حتى تظهر كلمة trycloudflare.com في السجلات
while ! grep -q "trycloudflare.com" cloudflare.log; do
    sleep 1
done

# نقرأ الرابط من السجلات
LINK=$(grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' cloudflare.log | head -n1)

echo "رابط النفق هو: $LINK"

# إبقاء العملية تعمل (اختياري)
wait $PID
