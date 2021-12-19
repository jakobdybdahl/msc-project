pip install .

echo "Hello UCloud..."

ps -ef | grep '/usr/local/bin/start-container' | grep -v grep | awk '{print $2}' | xargs -r kill -9