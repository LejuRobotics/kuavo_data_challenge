#!/bin/bash

# Store the arguments in variables
content="$1"
token="$2"

# Use the variables in the curl command
curl "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=$token" -H 'Content-Type: application/json' -d "
   {
    \"msgtype\": \"text\",
    \"text\": {
        \"content\": \"$content\"
    }
   }"